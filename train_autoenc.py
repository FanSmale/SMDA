import os
import torch
from tqdm import tqdm
from networks.autoencoder import AutoEnc
from fwi_dataset import FWIDataset
from torch.utils.data import DataLoader


def train_autoenc(model_config: dict = None):
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

    # Autoencoder setup
    auto_enc = AutoEnc(ch=model_config["channel"],
                        ch_mult=model_config["channel_mult"],
                        num_res_blocks=model_config["num_res_blocks"],
                        dropout=model_config["dropout"]).to(device)

    # Read existing autoencoder .pt files
    if model_config["training_load_weight"] is not None:
        auto_enc.load_state_dict(
            torch.load(
                os.path.join(model_config["save_weight_dir"], model_config["training_load_weight"]), map_location=device
            ),
            strict=False
        )
        print("Model weight load down.")

    optimizer = torch.optim.Adam(auto_enc.parameters(), lr=model_config["lr"])
    loss_criterion = torch.nn.MSELoss()

    for e in range(model_config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, _ in tqdmDataLoader:

                auto_enc.train()
                optimizer.zero_grad()

                x_0 = images.to(device)

                pred_x = auto_enc(x_0)

                loss = loss_criterion(pred_x, x_0)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    auto_enc.parameters(), model_config["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e + 1,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        torch.save(auto_enc.state_dict(),
                   os.path.join(model_config["save_weight_dir"],
                                model_config["save_model_name"] + str(e + 1) + ".pt"))


if __name__ == '__main__':

    openfwi_config = {
        "epoch": 200,
        "batch_size": 20,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 1,
        "dropout": 0.1,
        "lr": 1e-4,
        "img_size": 64,
        "grad_clip": 1.,
        "save_model_name": "autoenc",
        "device": "cuda:0",
        "training_load_weight": None,                   # When training for the first time, it is set to None
        "save_weight_dir": "./checkpoints/",
        "test_load_zsem": "autoenc.pt",
        "sampled_dir": "./sampled_imgs/"
    }
    train_autoenc(openfwi_config)
