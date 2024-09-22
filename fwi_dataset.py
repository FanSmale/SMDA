# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on May 2024

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pylab as plt
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Dataset
import torch.nn.functional as F


class FWIDataset(Dataset):
    """
    A series of operations on data
    """
    def __init__(self, config_path: str = "./configuration/config.json", training_strategy: str = None):
        """
        Constructor

        :param config_path:         The path of configuration
        :param training_strategy:   Force selection of the dataset to be read (opt.)
                                    By default, which dataset is read is based on the configuration.
        """

        with open(config_path) as f:
            cfg = json.load(f)

        self.data_cfg = cfg["datasets"]
        self.net_cfg = cfg["net_param"]

        if training_strategy is not None:
            self.net_cfg["training_strategy"] = training_strategy

        if self.net_cfg["training_strategy"] == "Vel_Fault":
            self.datasets_source = ["CVA", "CFA", "FFA"]
        elif self.net_cfg["training_strategy"] == "Marmousi":
            self.datasets_source = ["M20"]
        else:
            exit()

        self.data_volume = self.data_cfg[self.datasets_source[0]]["volume_size"]
        self.max_velo = self.data_cfg[self.datasets_source[0]]["max_velocity"]
        self.min_velo = self.data_cfg[self.datasets_source[0]]["min_velocity"]
        self.data_zip = []
        self.data_size = 0

        self.data_path_dict = self.determine_info(self.net_cfg["data_list_path"])

    def norm(self):
        """ Normalization """

        for i in range(len(self.data_zip[0])):
            for j in range(self.data_zip[0][i].shape[0]):
                self.data_zip[0][i][j][0] -= self.min_velo
                self.data_zip[0][i][j][0] /= (self.max_velo - self.min_velo)
                self.data_zip[0][i][j][0] = self.data_zip[0][i][j][0] * 2 - 1

    def determine_info(self, readtxt_dir: str = "./configuration/datalist.txt"):
        """
        Obtain the path to the data that needs to be read

        :param readtxt_dir:         Data list
        :return:                    List of stored path strings
        """

        temp_list = []
        with open(readtxt_dir, 'r') as file:
            for line in file:
                temp_list.append(line.strip())

        # Change the "training_strategy" in config.json to determine whether to read the dataset from A or B domain
        temp_dict = {}
        if self.net_cfg["training_strategy"] == "Vel_Fault":     # A Domain (from "Vel" and "Fault" families in OpenFWI)
            temp_dict.update({"cva": temp_list[temp_list.index("cva") + 1: temp_list.index("cva_end")]})
            temp_dict.update({"cfa": temp_list[temp_list.index("cfa") + 1: temp_list.index("cfa_end")]})
            temp_dict.update({"ffa": temp_list[temp_list.index("ffa") + 1: temp_list.index("ffa_end")]})
        elif self.net_cfg["training_strategy"] == "Marmousi":    # B Domain (20 slices of Marmousi)
            temp_dict.update({"m20": temp_list[temp_list.index("m20") + 1: temp_list.index("m20_end")]})
        else:
            exit()

        return temp_dict

    def load2memory(self):
        """
        After determining which and where data to read, start reading
        This operation may occupy a large amount of memory.
        """
        print("-----------------------------------")
        print("Loading datasets ...")
        self.data_zip = [[], []]

        for ind, dataset in enumerate(self.data_path_dict):     # Use 'ind' to represent categories
            print(dataset)
            path_list = self.data_path_dict[dataset]
            for k, path in enumerate(path_list):
                print(path.rstrip('\n'))
                temp = np.load(path.rstrip('\n'))
                temp2 = F.interpolate(torch.Tensor(temp), size=self.net_cfg["new_size"],
                                      mode='bilinear', align_corners=False).numpy()
                self.data_zip[0].append(temp2)
                self.data_zip[1].append([ind for k in range(self.data_volume)])
                self.data_size += self.data_volume
        print("Done!")
        print("-----------------------------------")
        self.norm()

    def __getitem__(self, idx: int = 0):
        """
        :param idx:     a value range from 0 to "data_size"
        :return:        a tuple of data
        """
        # idx = idx // 500
        batch_idx, sample_idx = idx // self.data_volume, idx % self.data_volume
        temp_data_zip = []
        for i in range(2):
            temp_data_zip.append(self.data_zip[i][batch_idx][sample_idx])

        return tuple(temp_data_zip)

    def __len__(self):
        """
        :return: Number of data and data files
        """
        return self.data_size

    def show_vmodel(self, idx: int = 0):
        """
        Display velocity model within the current FWIDataset class

        :param idx:         The position of the displayed data in the entire read dataset
        :return:
        """

        assert self.data_zip != [], "Please read the data into memory before presenting the data"

        batch_idx, sample_idx = idx // self.data_volume, idx % self.data_volume

        vmodel = self.data_zip[0][batch_idx][sample_idx][0]

        if np.max(vmodel) <= 1.0:
            vmodel = (vmodel + 1.0) / 2
            vmodel = vmodel * (self.max_velo - self.min_velo) + self.min_velo

        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

        im = ax.imshow(vmodel, extent=[0, 0.7, 0.7, 0], vmin=self.min_velo, vmax=self.max_velo)

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

        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(self.min_velo, self.max_velo, 7), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))

        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)
        plt.show()


def example_test():
    """
    An Example of using the FWIDataset class

    :return:
    """
    temp_dataset = FWIDataset(config_path="./configuration/config.json")
    temp_dataset.load2memory()

    p = 3
    temp_dataset.show_vmodel(idx=p)

    temp_data_zip = temp_dataset[p]
    vmodel = temp_data_zip[0]
    print(vmodel.shape)

    label = temp_data_zip[1]
    print(label)


if __name__ == '__main__':
    example_test()

