import os
import torch
from tqdm import tqdm

from diffusion import GaussianDiffusionTrainer
from networks.unet import UNet
from networks.autoencoder import AutoEnc
from fwi_dataset import FWIDataset
from torch.utils.data import DataLoader


def train_eps(model_config: dict):
    """

    :param model_config:            Configuration information
    :return:
    """

    device = torch.device(model_config["device"])

    vmodel_datasets = FWIDataset(config_path="./configuration/config.json")
    vmodel_datasets.load2memory()

    dataloader = DataLoader(
        vmodel_datasets, batch_size=model_config["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)

    # Network eps setup
    eps_model = UNet(T=model_config["T"],
                     n_class=model_config["num_class"],
                     ch=model_config["channel"],
                     ch_mult=model_config["channel_mult"],
                     num_res_blocks=model_config["num_res_blocks"],
                     dropout=model_config["dropout"]).to(device)

    # Autoencoder setup
    auto_enc = AutoEnc(ch=model_config["channel"],
                       ch_mult=model_config["channel_mult"],
                       num_res_blocks=model_config["num_res_blocks"],
                       dropout=model_config["dropout"]).to(device)

    # Read existing network eps .pt files (opt.)
    if model_config["training_load_weight"] is not None:
        eps_model.load_state_dict(
            torch.load(
                os.path.join(model_config["save_weight_dir"],
                             model_config["training_load_weight"]), map_location=device
            ),
            strict=False
        )
        print("Model weight load down.")

    # Read existing autoencoder .pt files
    auto_enc.load_state_dict(
        torch.load(
            os.path.join(model_config["save_weight_dir"], model_config["training_load_zsem"]), map_location=device
        ),
        strict=False
    )
    auto_enc.eval()
    print("Auto encoder weight load down.")

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=model_config["lr"])

    # Initialize diffusion model parameters
    trainer = GaussianDiffusionTrainer(eps_model,
                                       model_config["beta_1"],
                                       model_config["beta_T"],
                                       model_config["T"]).to(device)

    # Start training
    for e in range(model_config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:

                b = images.shape[0]
                optimizer.zero_grad()

                x_0 = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    zsem = auto_enc.encode(x_0).to(device)

                # There is only one category for Marmousi slices, so all category labels in a batch are set to 0.
                if vmodel_datasets.net_cfg["training_strategy"] == 'Marmousi':
                    labels = torch.zeros_like(labels).to(device)

                loss = trainer(x_0, labels, zsem).sum() / b ** 2.
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    eps_model.parameters(), model_config["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e + 1,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        torch.save(eps_model.state_dict(), os.path.join(model_config["save_weight_dir"],
                                                        model_config["save_model_name"] + str(e + 1) + ".pt"))


if __name__ == '__main__':

    openfwi_config = {
        "num_class": 3,
        "epoch": 200,
        "batch_size": 20,
        "T": 1000,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 1,
        "dropout": 0.1,
        "lr": 1e-4,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "save_model_name": "openfwi_eps",
        "device": "cuda:0",
        "training_load_zsem": "autoenc.pt",
        "training_load_weight": "openfwi_eps.pt",       # When training for the first time, it is set to None
        "save_weight_dir": "./checkpoints/"
    }

    m20_config = {
        "num_class": 3,
        "epoch": 200,
        "batch_size": 20,
        "T": 1000,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 1,
        "dropout": 0.1,
        "lr": 1e-4,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "save_model_name": "m20_eps",
        "device": "cuda:0",
        "training_load_zsem": "autoenc.pt",
        "training_load_weight": "m20_eps.pt",           # When training for the first time, it is set to None
        "save_weight_dir": "./checkpoints/"
    }
    train_eps(m20_config)
