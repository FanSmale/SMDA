import os
import numpy as np
import torch
from numpy.linalg import norm
import matplotlib.pylab as plt
import lpips
from diffusion import GaussianDiffusionSampler
from networks.unet import UNet
from networks.autoencoder import AutoEnc
from fwi_dataset import FWIDataset

percept_loss = lpips.LPIPS(net='vgg').cuda()


def lerp(x: torch.tensor, y: torch.tensor, t: float):
    """
    Linear interpolation

    :param x:       Input image A
    :param y:       Input image B
    :param t:       Interpolation parameter
    :return:        Fused image
    """
    return (1 - t) * x + t * y


def slerp(img1, img2, p):
    """
    Spherical linear interpolation

    :param img1:    Input image A
    :param img2:    Input image B
    :param p:       Interpolation parameter
    :return:        Fused image
    """
    # Flatten the images to vectors
    vec1 = img1.flatten().astype(np.float32)
    vec2 = img2.flatten().astype(np.float32)

    # Normalize the vectors to unit vectors
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    vec1_normalized = vec1 / norm1
    vec2_normalized = vec2 / norm2

    # Compute the dot product
    dot = np.dot(vec1_normalized, vec2_normalized)
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate theta
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        # If the vectors are almost parallel, use linear interpolation
        interp_vec = (1 - p) * vec1 + p * vec2
    else:
        theta = theta_0 * p
        sin_theta = np.sin(theta)
        sin_theta_t = np.sin(theta_0 - theta)

        s0 = sin_theta_t / sin_theta_0
        s1 = sin_theta / sin_theta_0

        interp_vec = s0 * vec1_normalized + s1 * vec2_normalized
        interp_vec = interp_vec * ((1 - p) * norm1 + p * norm2)

    # Reshape the interpolated vector back to image shape
    interp_img = interp_vec.reshape(img1.shape)
    return interp_img


def smda_for_single(model_config: dict):
    """
    Perform SMDA on a single sample pair

    :param model_config:                    Configuration information
    :return:
    """
    device = torch.device(model_config["device"])

    with torch.no_grad():

        # Preparation of slices of Marmousi (Domain B)
        m20_datasets = FWIDataset(config_path="./configuration/config.json", training_strategy="Marmousi")
        m20_datasets.load2memory()
        m20_ids = torch.tensor([9])     # Id from domain B (0 - 19)
        m20 = np.zeros([len(m20_ids), 1, *m20_datasets.net_cfg["new_size"]]).astype(np.float32)
        m20[0][0] = m20_datasets[m20_ids.tolist()[0]][0]
        # Domain B has only one category (c = {0})
        m20_labels = torch.ones(size=[model_config["batch_size"]], device=device).long() * 0
        m20 = torch.tensor(m20).to(device)

        # Preparation of OpenFWI (Domain A)
        opf_datasets = FWIDataset(config_path="./configuration/config.json")
        opf_datasets.load2memory()
        opf_ids = torch.tensor([11256])  # Id from domain A (CVA: 0 - 9999) (CFA: 10000 - 19999) (FFA: 20000 - 29999)
        opf = np.zeros([len(opf_ids), 1, *opf_datasets.net_cfg["new_size"]]).astype(np.float32)
        opf[0][0] = opf_datasets[opf_ids.tolist()[0]][0]
        # Domain A has three categories (c = {0, 1, 2})
        opf_labels = torch.ones(size=[model_config["batch_size"]], device=device).long() * (opf_ids[0].item() // 10000)
        opf = torch.tensor(opf).to(device)

        # Load $\epsilon^\textbf{B}_\theta$
        m20_eps_model = UNet(T=model_config["T"],
                             n_class=model_config["num_class"],
                             ch=model_config["channel"],
                             ch_mult=model_config["channel_mult"],
                             num_res_blocks=model_config["num_res_blocks"],
                             dropout=model_config["dropout"]).to(device)
        m20_eps_model.load_state_dict(torch.load(os.path.join(
            model_config["save_weight_dir"], "m20_eps.pt"), map_location=device))
        print(r"\epsilon^\textbf{B}_\theta model is loaded")

        # Load $\epsilon^\textbf{A}_\theta$
        opf_eps_model = UNet(T=model_config["T"],
                             n_class=model_config["num_class"],
                             ch=model_config["channel"],
                             ch_mult=model_config["channel_mult"],
                             num_res_blocks=model_config["num_res_blocks"],
                             dropout=model_config["dropout"]).to(device)
        opf_eps_model.load_state_dict(torch.load(os.path.join(
            model_config["save_weight_dir"], "openfwi_eps.pt"), map_location=device))
        print(r"\epsilon^\textbf{A}_\theta model is loaded")

        # Load autoencoder
        auto_enc = AutoEnc(ch=model_config["channel"],
                           ch_mult=model_config["channel_mult"],
                           num_res_blocks=model_config["num_res_blocks"],
                           dropout=model_config["dropout"]).to(device)
        auto_enc.load_state_dict(torch.load(os.path.join(
            model_config["save_weight_dir"], "autoenc.pt"), map_location=device))
        print("Autoencoder eps model is loaded")

        m20_sampler = GaussianDiffusionSampler(model=m20_eps_model,
                                               beta_1=model_config["beta_1"],
                                               beta_T=model_config["beta_T"],
                                               T=model_config["T"],
                                               section_counts=model_config["section_counts"]).to(device)
        m20_zsem = auto_enc.encode(m20.detach().clone()).to(device)

        opf_sampler = GaussianDiffusionSampler(model=opf_eps_model,
                                               beta_1=model_config["beta_1"],
                                               beta_T=model_config["beta_T"],
                                               T=model_config["T"],
                                               section_counts=model_config["section_counts"]).to(device)
        opf_zsem = auto_enc.encode(opf.detach().clone()).to(device)

        # DDIM encoding process [x_0 -> x_T]
        m20_noise = m20_sampler.ddim_reverse_sample_loop(forward_T=model_config["forward_T"],
                                                         x_0=m20,
                                                         n_class=m20_labels,
                                                         zsem=m20_zsem)
                                                         # save_gif_path=r".\sampled_imgs\Marmousi_resample.gif")
        opf_noise = opf_sampler.ddim_reverse_sample_loop(forward_T=model_config["forward_T"],
                                                         x_0=opf,
                                                         n_class=opf_labels,
                                                         zsem=opf_zsem)
                                                         # save_gif_path=r".\sampled_imgs\OpenFWI_resample.gif")

        # Sample enumeration in the range of p
        for ind, p in enumerate(np.linspace(0.0, 1.0, 150)):

            temp_m20_noise = m20_noise.clone()
            temp_opf_noise = opf_noise.clone()
            temp_m20_zsem = m20_zsem.clone()
            temp_opf_zsem = opf_zsem.clone()

            # Interpolation fusion
            temp_zsem = lerp(temp_opf_zsem, temp_m20_zsem, p)
            temp_noise = torch.tensor(slerp(temp_opf_noise.cpu().numpy(), temp_m20_noise.cpu().numpy(), p)).to(device)

            # DDIM decoding process (guided by $\epsilon^\textbf{B}_\theta$) [x_T -> x_0]
            sampled_imgs = m20_sampler.ddim_sample_loop(backend_T=model_config["backend_T"],
                                                        x_t=temp_noise,
                                                        n_class=m20_labels,
                                                        zsem=temp_zsem)

            # DDIM decoding process (guided by $\epsilon^\textbf{A}_\theta$) [x_T -> x_0]
            # sampled_imgs = opf_sampler.ddim_sample_loop(backend_T=model_config["backend_T"],
            #                                             x_t=temp_noise,
            #                                             n_class=opf_labels,
            #                                             zsem=temp_zsem,
            #                                             save_gif_path=None)

            A_loss = percept_loss(opf[0][0], sampled_imgs[0][0]).item()
            B_loss = percept_loss(m20[0][0], sampled_imgs[0][0]).item()

            A = opf[0][0].cpu().detach().numpy()
            B = m20[0][0].cpu().detach().numpy()
            generated_v = sampled_imgs[0][0].cpu().detach().numpy()

            # Save to local (The first way)
            # _, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=300, sharex=True, sharey=True)
            # ax.imshow(generated_v)
            # plt.tight_layout()
            # plt.savefig(os.path.join(model_config["sampled_dir"], 'm{}.png'.format(abs(A_loss-B_loss))))
            # plt.close()

            # Save to local (The second way)
            _, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150, sharex=True, sharey=True)
            ax[0].imshow(A, vmin=-1., vmax=1.)
            ax[0].set_title("A\nLPIPS={:.4f}".format(A_loss))
            ax[1].imshow(generated_v, vmin=-1., vmax=1.)
            ax[1].set_title("Between A and B (t={:.4f})\nall_loss={:.4f} diff_loss={:.4f}".format(p, A_loss+B_loss, abs(A_loss-B_loss)))
            ax[2].imshow(B, vmin=-1., vmax=1.)
            ax[2].set_title("B\nLPIPS={:.4f}".format(B_loss))
            plt.tight_layout()
            plt.savefig(os.path.join(model_config["sampled_dir"], 'm{}.png'.format(ind)))
            np.save(os.path.join(model_config["sampled_dir"], 'm{}.npy'.format(ind)), generated_v)
            plt.close()


if __name__ == '__main__':
    config = {
        "num_class": 3,
        "batch_size": 1,                            # Here, batch size must be one
        "T": 1000,
        "section_counts": None,
        "forward_T": 80,
        "backend_T": 80,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 1,
        "dropout": 0.1,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "device": "cuda:0",
        "save_weight_dir": "./checkpoints/",
        "sampled_dir": "./sampled_imgs/"
    }
    smda_for_single(config)
