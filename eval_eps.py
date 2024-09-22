import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pylab as plt
from diffusion import GaussianDiffusionSampler
from networks.unet import UNet
from networks.autoencoder import AutoEnc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fwi_dataset import FWIDataset


def show_vmodel(vmodel, save_path=None):

    min_velo = np.min(vmodel)
    max_velo = np.max(vmodel)

    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

    im = ax.imshow(vmodel, extent=[0, 0.7, 0.7, 0], vmin=min_velo, vmax=max_velo)

    font18 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    plt.rcParams['font.size'] = 14
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.35)

    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', ticks=np.linspace(min_velo, max_velo, 7),
                 format=mpl.ticker.StrMethodFormatter('{x:.0f}'))

    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def eval_eps(model_config: dict):
    """
    Evaluate the effect of the trained network $\epsilon_\theta$

    :param model_config:        Configuration information
    :return:
    """

    device = torch.device(model_config["device"])

    with torch.no_grad():

        eps_model = UNet(T=model_config["T"],
                         n_class=model_config["num_class"],
                         ch=model_config["channel"],
                         ch_mult=model_config["channel_mult"],
                         num_res_blocks=model_config["num_res_blocks"],
                         dropout=model_config["dropout"]).to(device)

        auto_enc = AutoEnc(ch=model_config["channel"],
                           ch_mult=model_config["channel_mult"],
                           num_res_blocks=model_config["num_res_blocks"],
                           dropout=model_config["dropout"]).to(device)

        eps_model.load_state_dict(
            torch.load(
                os.path.join(model_config["save_weight_dir"], model_config["test_load_weight"]),
                map_location=device)
        )

        auto_enc.load_state_dict(
            torch.load(
                os.path.join(model_config["save_weight_dir"], model_config["test_load_zsem"]),
                map_location=device)
        )

        eps_model.eval()
        auto_enc.eval()

        openfwi_inner_smda(eps_model, auto_enc, model_config)


def openfwi_inner_smda(eps_model, auto_enc, model_config):
    """
    Fusion of different categories within OpenFWI.

    Note that this approach does not involve interpolation fusion.
    We first encode the velocity model and then directly encode it with other categories.


    :param eps_model:           Trained network $\epsilon_\theta$
    :param auto_enc:            Automatic encoder
    :param model_config:
    :return:
    """

    temp_dataset = FWIDataset(config_path="./configuration/config.json")
    temp_dataset.load2memory()

    device = torch.device(model_config["device"])

    # If the both lists here are exactly the same, then this
    # experiment becomes a complete reconstruction example of DDIM.

    cva_ids = [12, 34, 77, 67, 82]                  # Optional id between 0 and 9999 (cva = CurveVelA)
    cfa_ids = [10012, 10034, 10077, 10067, 10082]   # Optional id between 10000 and 19999 (cfa = CurveFaultA)

    assert len(cva_ids) == len(cfa_ids) == model_config["batch_size"]

    cva = np.zeros([len(cva_ids), 1, *temp_dataset.net_cfg["new_size"]]).astype(np.float32)
    for seq_i, id in enumerate(cva_ids):
        cva[seq_i][0] = temp_dataset[id][0]
    # The category of CurveVelA is 0
    cva_labels = torch.ones(size=[model_config["batch_size"]], device=device).long() * 0

    cfa = np.zeros([len(cfa_ids), 1, *temp_dataset.net_cfg["new_size"]]).astype(np.float32)
    for seq_i, id in enumerate(cfa_ids):
        cfa[seq_i][0] = temp_dataset[id][0]
    # The category of CurveVelA is 1
    cfa_labels = torch.ones(size=[model_config["batch_size"]], device=device).long() * 1

    with torch.no_grad():

        sampler = GaussianDiffusionSampler(model=eps_model,
                                           beta_1=model_config["beta_1"],
                                           beta_T=model_config["beta_T"],
                                           T=model_config["T"],
                                           section_counts=model_config["section_counts"]).to(device)

        cva_zsem = auto_enc.encode(torch.tensor(cva).to(device)).to(device)
        cfa_zsem = auto_enc.encode(torch.tensor(cfa).to(device)).to(device)

        # x_0 -> x_T
        noise = sampler.ddim_reverse_sample_loop(forward_T=model_config["forward_T"],
                                                 x_0=torch.tensor(cfa).to(device),
                                                 n_class=cfa_labels,
                                                 zsem=cfa_zsem,
                                                 save_gif_path=os.path.join(model_config["sampled_dir"],
                                                                            'ddim_reverse_sample.gif')
                                                 )
        # x_T -> x_0
        sampled_imgs = sampler.ddim_sample_loop(backend_T=model_config["backend_T"],
                                                x_t=noise,
                                                n_class=cva_labels,
                                                zsem=cva_zsem,
                                                save_gif_path=os.path.join(model_config["sampled_dir"],
                                                                           'ddim_sample.gif')
                                                )

        for i in range(model_config["batch_size"]):
            generate_v = sampled_imgs[i][0].cpu().detach().numpy()
            show_vmodel(generate_v, save_path=os.path.join(model_config["sampled_dir"], 'test_vm{}.png'.format(i)))


if __name__ == '__main__':

    openfwi_config = {
        "num_class": 3,
        "section_counts": None,
        "batch_size": 5,
        "T": 1000,
        "forward_T": 200,
        "backend_T": 200,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 1,
        "dropout": 0.1,
        "lr": 1e-4,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "openfwi_eps.pt",
        "test_load_zsem": "autoenc.pt",
        "sampled_dir": "./sampled_imgs/"
    }
    eval_eps(openfwi_config)


