import os
import numpy as np
import torch
import matplotlib.pylab as plt
from networks.autoencoder import AutoEnc
from fwi_dataset import FWIDataset


def eval_eps(model_config: dict):
    """
    Evaluate the effect of the trained autoencoder

    :param model_config:        Configuration information
    :return:
    """

    device = torch.device(model_config["device"])

    # Select openfwi dataset
    temp_dataset = FWIDataset(config_path="./configuration/config.json")
    temp_dataset.load2memory()
    selected_ids = [1150, 11150, 21000, 250, 10018]
    assert len(selected_ids) == model_config["batch_size"]
    v = np.zeros([len(selected_ids), 1, *temp_dataset.net_cfg["new_size"]]).astype(np.float32)
    for seq_i, id in enumerate(selected_ids):
        v[seq_i][0] = temp_dataset[id][0]
    v = torch.tensor(v).to(device)

    # load model and evaluate
    with torch.no_grad():

        auto_enc = AutoEnc(ch=model_config["channel"],
                           ch_mult=model_config["channel_mult"],
                           num_res_blocks=model_config["num_res_blocks"],
                           dropout=model_config["dropout"]).to(device)

        auto_enc.load_state_dict(
            torch.load(
                os.path.join(model_config["save_weight_dir"], model_config["test_load_zsem"]),
                map_location=device)
        )

        auto_enc.eval()
        minor_v = auto_enc(v)

        loss_criterion = torch.nn.MSELoss()
        loss = loss_criterion(minor_v, v)

        print(loss.item())

        for seq_i, id in enumerate(selected_ids):
            _, ax = plt.subplots(2, figsize=(10.5, 10.5), sharex=True, sharey=True)
            ax[0].imshow(v[seq_i][0].cpu())
            ax[0].set_title("Input")
            ax[1].imshow(minor_v[seq_i][0].detach().cpu())
            ax[1].set_title("Output")
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    model_config = {
        "batch_size": 5,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 1,
        "dropout": 0.1,
        "img_size": 64,
        "device": "cuda:0",
        "save_weight_dir": "./checkpoints/",
        "test_load_zsem": "autoenc.pt"
    }
    eval_eps(model_config)

