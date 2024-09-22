import os
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
import gc
import random
import matplotlib.pylab as plt
import lpips
from diffusion import GaussianDiffusionSampler
from networks.unet import UNet
from networks.autoencoder import AutoEnc
from fwi_dataset import FWIDataset


percept_loss = lpips.LPIPS(net='vgg').cuda()


def tensor_del_rows(rows_to_delete: list, data: torch.tensor):
    """
    Batch delete elements in tensor list

    :param rows_to_delete:          List of indices of elements to be deleted
    :param data:                    Tensor list
    :return:
    """
    rows_to_keep = [i for i in range(data.size(0)) if i not in rows_to_delete]
    return data[rows_to_keep, :]


class BatchControl(nn.Module):
    """
    Used to operate on data within a batch
    """
    def __init__(self, para_batch_num: int, para_init_vmodels: torch.Tensor, para_vmodels: torch.Tensor,
                 para_labels: torch.Tensor, para_zsem: torch.Tensor, para_ids: list):
        """

        :param para_batch_num:      The amount of data in the current batch
        :param para_init_vmodels:   x^D_0
        :param para_vmodels:        x^D_T
        :param para_labels:         List of category information
        :param para_zsem:           Semantic latent code
        :param para_ids:            Data id list
        """
        super().__init__()

        self.__batch_num = para_batch_num
        self.__init_vmodels = para_init_vmodels.clone()
        self.__vmodels = para_vmodels.clone()
        self.__labels = para_labels.clone()
        self.__zsem = para_zsem.clone()
        self.__lpips_change_list = None
        self.__ids = torch.tensor(para_ids).long().unsqueeze(1)

    def get_batch_num(self):
        """
        getter

        :return:                    The amount of data in the current batch
        """
        return self.__batch_num

    def resize_batch_num(self, para_batch_num):
        """
        setter

        :param para_batch_num:      Reset the amount of data in the current batch
        :return:
        """
        self.__batch_num = para_batch_num

    def get_init_vmodels(self, is_numpy=False):
        """
        getter

        :param is_numpy:            Whether to output as ndarray type
        :return:                    x^D_0
        """
        if is_numpy:
            return self.__init_vmodels.cpu().numpy()
        return self.__init_vmodels

    def get_vmodel(self, is_numpy=False):
        """
        getter

        :param is_numpy:            Whether to output as ndarray type
        :return:                    x^D_T
        """
        if is_numpy:
            return self.__vmodels.cpu().numpy()
        return self.__vmodels

    def get_zsem(self, is_numpy=False):
        """
        getter

        :param is_numpy:            Whether to output as ndarray type
        :return:                    Semantic latent code
        """
        if is_numpy:
            return self.__zsem.cpu().numpy()
        return self.__zsem

    def get_labels(self, is_numpy=False):
        """
        getter

        :param is_numpy:            Whether to output as ndarray type
        :return:                    List of category information
        """
        if is_numpy:
            return self.__labels.cpu().numpy()
        return self.__labels

    def get_ids(self, is_numpy=False):
        """
        getter

        :param is_numpy:            Whether to output as ndarray type
        :return:                    Data id list
        """
        if is_numpy:
            return self.__ids.view(-1).numpy()
        return self.__ids.view(-1)

    def get_id(self, index: int):
        """
        getter

        :param index:               Index of data id list
        :return:                    Data id
        """
        return self.__ids[index].item()

    def del_eles(self, del_list: list):
        """
        Batch delete data from the current batch according to the index list

        :param del_list:            Index list for deleting
        :return:
        """

        assert self.__batch_num >= len(del_list)

        self.__init_vmodels = tensor_del_rows(del_list, self.__init_vmodels)
        self.__vmodels = tensor_del_rows(del_list, self.__vmodels)
        self.__labels = self.__labels[len(del_list):]       # 目前暂未针对全部分类进行随机融合, 因此标签都是一样的
        self.__zsem = tensor_del_rows(del_list, self.__zsem)
        self.__ids = tensor_del_rows(del_list, self.__ids)


class SMDA(nn.Module):
    """
    SMDA: Velocity Profile Synthesis in Multi-Domain using Diffusion Autoencoders
    """
    def __init__(self, config: dict, opf_ids: list, m20_ids: list, opf_type: str = "cfa"):
        """
        constructor

        :param config:              Configuration information
        :param opf_ids:             The ID table of domain A (OpenFWI) in the current batch
        :param m20_ids:             The ID table of domain B (Marmousi) in the current batch
        :param opf_type:            The category of domain A data
                                    (domain B data only has one type, so there is no need to set it)
        """
        super().__init__()

        type2labels = {"cva": 0, "cfa": 1, "ffa": 2}
        self.config = config
        self.batch_size = self.config["batch_size"]
        assert self.batch_size == len(opf_ids) == len(m20_ids)
        self.device = torch.device(self.config["device"])
        self.paired_records = set()
        self.opf_dataset = FWIDataset(config_path="./configuration/config.json")
        self.opf_dataset.load2memory()
        self.m20_dataset = FWIDataset(config_path="./configuration/config.json", training_strategy="Marmousi")
        self.m20_dataset.load2memory()

        # Initialization for domain A
        self.xT_opf = None
        self.opf_sampler = None
        self.opf_zsem = None
        self.opf_labels = None
        self.opf_ids = opf_ids

        # Initialization for domain B
        self.xT_m20 = None
        self.m20_sampler = None
        self.m20_zsem = None
        self.m20_labels = None
        self.m20_ids = m20_ids

        with torch.no_grad():

            # Initialization for Autoencoder
            auto_enc = AutoEnc(ch=self.config["channel"], ch_mult=self.config["channel_mult"],
                               num_res_blocks=self.config["num_res_blocks"], dropout=self.config["dropout"]).to(self.device)
            auto_enc.load_state_dict(torch.load(os.path.join(self.config["read_weight_dir"],
                                                             self.config["auto_encoder_name"]), map_location=self.device))

            # Initialization for x_0
            self.v_opf = np.zeros([self.batch_size, 1, *self.opf_dataset.net_cfg["new_size"]], dtype=np.float32)
            self.v_m20 = np.zeros([self.batch_size, 1, *self.m20_dataset.net_cfg["new_size"]], dtype=np.float32)
            for seq_i, (opf_id, m20_id) in enumerate(zip(self.opf_ids, self.m20_ids)):
                self.v_opf[seq_i][0] = self.opf_dataset[opf_id][0]
                self.v_m20[seq_i][0] = self.m20_dataset[m20_id][0]
            self.v_opf = torch.tensor(self.v_opf).to(self.device)
            self.v_m20 = torch.tensor(self.v_m20).to(self.device)

            # Get sampler (DDIM) and semantic latent code z_sem
            self.opf_sampler, self.opf_zsem = self.get_sampler_and_zsem(init_vmodel=self.v_opf,
                                                                        auto_enc=auto_enc,
                                                                        eps_name=self.config["openfwi_eps_name"])
            self.m20_sampler, self.m20_zsem = self.get_sampler_and_zsem(init_vmodel=self.v_m20,
                                                                        auto_enc=auto_enc,
                                                                        eps_name=self.config["marmousi_eps_name"])

            # Get labels
            self.opf_labels = torch.ones(self.batch_size, device=self.device).long() * type2labels[opf_type]
            self.m20_labels = torch.zeros(self.batch_size, device=self.device).long()

            # Get x_T (encoding DDIM)
            self.xT_opf = self.opf_sampler.ddim_reverse_sample_loop(forward_T=self.config["forward_T"],
                                                                    x_0=self.v_opf,
                                                                    n_class=self.opf_labels,
                                                                    zsem=self.opf_zsem,
                                                                    use_tqdm=True)

            self.xT_m20 = self.m20_sampler.ddim_reverse_sample_loop(forward_T=self.config["forward_T"],
                                                                    x_0=self.v_m20,
                                                                    n_class=self.m20_labels,
                                                                    zsem=self.m20_zsem,
                                                                    use_tqdm=True)

    @classmethod
    def lerp(cls, x: torch.tensor, y: torch.tensor, p: float):
        """
        Linear interpolation

        :param x:       Input image A
        :param y:       Input image B
        :param p:       Interpolation parameter p
        :return:        Fused image
        """
        return (1 - p) * x + p * y

    @classmethod
    def slerp(cls, img1, img2, p):
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

    def get_sampler_and_zsem(self, init_vmodel: torch.tensor, auto_enc: AutoEnc, eps_name: str):
        """
        Get sampler (DDIM) and semantic latent code z_sem

        :param init_vmodel:
        :param auto_enc:
        :param eps_name:
        :return:
        """

        eps = UNet(T=self.config["T"], n_class=self.config["num_class"], ch=self.config["channel"],
                   ch_mult=self.config["channel_mult"], num_res_blocks=self.config["num_res_blocks"],
                   dropout=self.config["dropout"]).to(self.device)
        eps.load_state_dict(torch.load(os.path.join(self.config["read_weight_dir"], eps_name), map_location=self.device))

        zsem = auto_enc.encode(init_vmodel.detach())

        sampler = GaussianDiffusionSampler(model=eps, beta_1=self.config["beta_1"], beta_T=self.config["beta_T"],
                                           T=self.config["T"],
                                           section_counts=self.config["section_counts"]).to(self.device)
        return sampler, zsem

    def pain(self, A_img, B_img, I_img, diff_loss, p, save_name):
        """
        pain results

        :param A_img:       Image from domain A
        :param B_img:       Image from domain B
        :param I_img:       Image from domain I
        :param diff_loss:    |L^A_\text{lp} - L^B_\text{lp}|
        :param p:           Interpolation parameter p
        :param save_name:   File name used for saving
        :return:
        """
        _, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
        ax[0].imshow(A_img, vmin=-1., vmax=1.)
        ax[0].set_title("OpenFWI")
        ax[1].imshow(I_img, vmin=-1., vmax=1.)
        ax[1].set_title("Between OpenFWI and Marmousi (t={:.4f})\ndiff_loss={:.4f}".format(p, diff_loss))
        ax[2].imshow(B_img, vmin=-1., vmax=1.)
        ax[2].set_title("Marmousi")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["sampled_dir"], "png/", save_name))
        plt.close()

    def batch_synthesis_module(self, p_range: list, accuracy: int = 60, threshold1: float = 0.001, threshold2: float = 0.005):
        """
        It is used to regulate the decoding process of batches of data and control overhead

        :param p_range:     The range of interpolation parameter fluctuations
        :param accuracy:    The number of sample enumerations within the interpolation parameter range
        :param threshold1:  $\theta_1$
        :param threshold2:  $\theta_2$
        :return:
        """

        # Get encoded dataset $\mathbf{E} = \{e_i\}^B_{i=1}$
        opf_bc = BatchControl(para_batch_num=self.batch_size,
                              para_init_vmodels=self.v_opf,
                              para_vmodels=self.xT_opf,
                              para_labels=self.opf_labels,
                              para_zsem=self.opf_zsem,
                              para_ids=self.opf_ids)
        m20_bc = BatchControl(para_batch_num=self.batch_size,
                              para_init_vmodels=self.v_m20,
                              para_vmodels=self.xT_m20,
                              para_labels=self.m20_labels,
                              para_zsem=self.m20_zsem,
                              para_ids=self.m20_ids)
        sample_num = self.batch_size

        iter_p_range = np.linspace(p_range[0], p_range[1], accuracy)
        # Setting up checkpoints
        check_p = [iter_p_range[int(accuracy * 0.3)],
                   iter_p_range[int(accuracy * 0.5)],
                   iter_p_range[int(accuracy * 0.65)],
                   iter_p_range[int(accuracy * 0.75)],
                   iter_p_range[int(accuracy * 0.8)]]

        # LPIPS dynamic change record table
        lpips_change_list = torch.ones([sample_num, accuracy])

        # Record the synthetic image of each data within a batch when the absolute difference of loss is the lowest
        minimum_diffloss_sampled_imgs = torch.zeros_like(self.v_opf)

        # Store the output log of the current batch in string
        log_txt = ""

        with torch.no_grad():

            # Explore all possible interpolation parameters
            for ind, p in enumerate(iter_p_range):
                print("\n********t: {:.4f}, batch: {}**************".format(p, sample_num)
                      + ("[Checkpoints]" if p in check_p else ""))
                print("Log: {}".format(log_txt))

                # Get z^I_{sem} and x^I_T corresponding to the interpolation parameter (in the form of batch data)
                mix_zsem = self.lerp(opf_bc.get_zsem(), m20_bc.get_zsem(), p)
                mix_noise = torch.zeros_like(opf_bc.get_vmodel()).to(self.device)
                for i in range(sample_num):
                    mix_noise[i][0] = torch.tensor(self.slerp(opf_bc.get_vmodel(True)[i][0],
                                                              m20_bc.get_vmodel(True)[i][0],
                                                              p)).to(self.device)
                # Here, x^I_T and x^I_0 are both batch-calculated in advance.
                # They do not need to be calculated one by one.
                print("Sampling...")
                sampled_imgs = self.m20_sampler.ddim_sample_loop(backend_T=self.config["backend_T"],
                                                                 x_t=mix_noise,
                                                                 n_class=m20_bc.get_labels(),
                                                                 zsem=mix_zsem)
                assert len(sampled_imgs) == sample_num

                del_dict = dict()                    # Store data that needs to be deleted {id: interpolation parameter}
                # Check all generated images in a batch one by one (filter)
                for s in range(sample_num):

                    opf_loss = percept_loss(opf_bc.get_init_vmodels()[s][0], sampled_imgs[s][0]).item()
                    m20_loss = percept_loss(m20_bc.get_init_vmodels()[s][0], sampled_imgs[s][0]).item()
                    diff_loss = abs(opf_loss - m20_loss)
                    lpips_change_list[s][ind] = diff_loss
                    print("diff_loss: {:.5f}".format(diff_loss))
                    if diff_loss == lpips_change_list[s].min():
                        minimum_diffloss_sampled_imgs[s][0] = sampled_imgs[s][0].cpu()

                    # diff_loss is lower than $\theta_1$, output directly
                    if diff_loss < threshold1:
                        del_dict.update({s: sampled_imgs[s][0].cpu()})
                    # If at the checkpoint
                    elif p in check_p:
                        # If the current diff_loss is higher than the historical minimum
                        if diff_loss > lpips_change_list[s].min():
                            # This difference is small enough, indicating that it may still fall -> it is fluctuating
                            if diff_loss - lpips_change_list[s].min() < 0.05:
                                pass    # Keep watching
                            # If the difference is too large, you need to consider deleting this data.
                            else:
                                del_dict.update({s: None})
                                # If it has ever been lower than $theta_2$, then this data is also acceptable output
                                if lpips_change_list[s].min() < threshold2:
                                    del_dict[s] = minimum_diffloss_sampled_imgs[s][0]
                        # If the current value is the lowest in history -> it is declining
                        else:
                            pass        # Keep watching
                print("Sampling completed!")

                if del_dict:
                    for batch_id in del_dict:
                        # It can synthesize synthetic data that meets expectations
                        if del_dict[batch_id] is not None:
                            # Save .png
                            save_name = "{}_{}".format(opf_bc.get_id(batch_id), m20_bc.get_id(batch_id))
                            self.pain(A_img=opf_bc.get_init_vmodels(True)[batch_id][0],
                                      B_img=m20_bc.get_init_vmodels(True)[batch_id][0],
                                      I_img=minimum_diffloss_sampled_imgs[batch_id][0].cpu(),
                                      diff_loss=lpips_change_list[batch_id].min(), p=p,
                                      save_name=save_name)
                            # Save .npy
                            np.save(os.path.join(self.config["sampled_dir"], "npy/{}.npy".format(save_name)),
                                    minimum_diffloss_sampled_imgs[batch_id].cpu().numpy())

                            # Log record (keyword: 0)
                            log_txt += (', "{}_{}"'.format(save_name, 0))

                        # It cannot synthesize synthetic data that meets expectations -> Need to delete in advance
                        else:
                            # Log record (keyword: 1)
                            log_txt += (', "{}_{}_{}"'.format(opf_bc.get_id(batch_id), m20_bc.get_id(batch_id), 1))

                    # In batch data, data for which suitable interpolation parameters have been found is deleted
                    # to improve the efficiency of the next decoding.
                    opf_bc.del_eles(list(del_dict.keys()))
                    m20_bc.del_eles(list(del_dict.keys()))
                    lpips_change_list = tensor_del_rows(list(del_dict.keys()), lpips_change_list)
                    minimum_diffloss_sampled_imgs = tensor_del_rows(list(del_dict.keys()), minimum_diffloss_sampled_imgs)

                    print("The decision of p={:.4f} is completed, and the number of batches decreases from {} to {}"
                          .format(p, sample_num, sample_num - len(del_dict)))
                    sample_num -= len(del_dict)
                    opf_bc.resize_batch_num(sample_num)
                    m20_bc.resize_batch_num(sample_num)
                else:
                    print("The decision of p={:.4f} is completed, and the number of batches does not change".format(p))

                if sample_num == 0:
                    print("All samples in the current batch have found the best sampling, ended early, "
                          "and the traversal ratio is {:.2f}".format(ind / accuracy))
                    break

            # If some data cannot be output in the end, their global minimum points are not covered.
            # Although these data are not output, they still need to be recorded (keyword: 2).
            if sample_num != 0:
                for i in range(sample_num):
                    log_txt += (', "{}_{}_{}"'.format(opf_bc.get_id(i), m20_bc.get_id(i), 2))

            # Logs written to local records
            with open('.\configuration\paired_records.txt', 'a') as file:
                file.write(log_txt)
                file.close()


def for_all_cfa_sampling(config: dict):
    """
    This sampling is to traverse all the data from the CurveFaultA (CFA) dataset and randomly match it with a data from Marmousi.
    The purpose of this function is to ensure that all data from CurveFaultA are fused.
    However, the number of synthesized data after execution is likely to be less than 10,000.

    :param config:          Configuration information
    :return:
    """

    while True:
        # Reading local records
        with open('.\configuration\paired_records.txt', 'r') as file:
            content = file.read()
            paired_records = eval(content)
            file.close()

        storing_opf_ids = []
        # storing_m20_ids = []
        if len(paired_records) != 0:
            storing_opf_ids = [int(item.split("_")[0]) for item in paired_records]
            # storing_m20_ids = [int(item.split("_")[1]) for item in paired_records]

        available_ids = list(set(range(10000, 19999)) - set(storing_opf_ids))
        if len(available_ids) == 0:
            break

        # The remaining samples are smaller than the set batch size
        if config["batch_size"] > len(available_ids):
            config["batch_size"] = len(available_ids)

        opf_ids = available_ids[:config["batch_size"]]
        m20_ids = torch.randint(20, size=(config["batch_size"],)).tolist()

        # Encoding
        generator = SMDA(config=config,
                         opf_ids=opf_ids,
                         m20_ids=m20_ids,
                         opf_type="cfa")
        # Multiple decoding
        generator.batch_synthesis_module(p_range=[0.0, 0.6], accuracy=80, threshold1=0.003, threshold2=0.01)

        del generator
        gc.collect()


def random_generate(config):
    """
    Both CurveFaultA and Marmousi slices are randomly selected.
    However, to ensure that there is a certain difference between the generated data.
    We artificially define some similarity categories for Marmousi slices.
    We hope to avoid duplication with previous data each time we randomly select Marmousi slice data.

    :param config:          Configuration information
    :return:
    """

    # Artificial classification of Marmousi slices
    simlar_lst = [[0, 1],
                  [2, 3, 4, 8, 9, 12, 15, 16, 18, 19],
                  [5, 13],
                  [6, 7, 10, 11, 14, 17]]
    simlar_dict = dict()
    for class_id, item_lst in enumerate(simlar_lst):
        for item in item_lst:
            simlar_dict.update({item: class_id})

    while True:
        # Reading local records "11262_13_0", "11262_13_0"
        with open('.\configuration\paired_records.txt', 'r') as file:

            paired_records = dict()  # CurveFaultA (id) -> Marmousi slices (id list)
            content = file.read()

            if content != '' and content.count(",") != 0:
                temp_lst = list(eval(content))
                for item in temp_lst:
                    if int(item.split("_")[0]) not in paired_records:
                        paired_records.update({int(item.split("_")[0]): [int(item.split("_")[1])]})
                    else:
                        paired_records[int(item.split("_")[0])].append(int(item.split("_")[1]))

            file.close()

        opf_ids = (torch.randint(10000, size=(config["batch_size"],)) + 10000).tolist()
        m20_ids = torch.randint(20, size=(config["batch_size"],)).tolist()

        del_index = []
        for index, (a_id, b_id) in enumerate(zip(opf_ids, m20_ids)):

            # When Flag is True, it means that the current ID pair needs to be replaced.
            # When Flag is False, it means that the current ID does not need to be replaced
            # or need to be deleted directly.
            flag = False

            temp_a_id = a_id
            temp_b_id = b_id
            history_b_id_list = []

            if temp_a_id in paired_records:
                history_b_id_list = paired_records[temp_a_id]

            # If the current data from CurveFaultA has been fused with all Marmousi categories, it will not be used.
            if len(history_b_id_list) == len(simlar_lst):
                del_index.append(index)
                continue

            history_b_id_class_set = set([simlar_dict[i] for i in history_b_id_list])

            while simlar_dict[temp_b_id] in history_b_id_class_set:
                temp_b_id = random.randint(0, 19)
                flag = True
            if flag:
                m20_ids[index] = temp_b_id
                paired_records[a_id].append(temp_b_id)

        if len(del_index) != 0:
            print("{} domain A data have been trained to saturation. "
                  "After they are removed, the batch size decreases.".format(len(del_index)))

            def delete_elements_by_indices(data: torch.tensor, indices: list):
                """
                Remove multiple elements from a list

                :param data:                Data List
                :param indices:             List of indices of elements to be deleted
                :return:                    Deleted data list
                """
                indices_set = set(indices)
                new_data = [item for idx, item in enumerate(data) if idx not in indices_set]
                return new_data

            opf_ids = delete_elements_by_indices(opf_ids, del_index)
            m20_ids = delete_elements_by_indices(m20_ids, del_index)
            config["batch_size"] = config["batch_size"] - len(del_index)

        print("opf_ids:\n{}".format(opf_ids))
        print("m20_ids:\n{}".format(m20_ids))

        generator = SMDA(config=config,
                         opf_ids=opf_ids,
                         m20_ids=m20_ids,
                         opf_type="cfa")
        # get x0
        generator.batch_synthesis_module(p_range=[0.0, 0.6], accuracy=80, threshold1=0.003, threshold2=0.01)

        del generator
        gc.collect()


if __name__ == '__main__':
    """
    日志文件解释: {A}_{B}_{C} A表示CFA文件, B表示M20文件,
    C表示匹配状态[0成功匹配|1曲线趋势不正确|2不能在指定区间下降|3有与之相似的另一种匹配(需要通过后期数据处理实现)|4有与之相似的另一种匹配(需要通过后期数据处理实现)]
    """
    config = {
        "num_class": 3,
        "batch_size": 30,
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
        "auto_encoder_name": "autoenc.pt",
        "openfwi_eps_name": "openfwi_eps.pt",
        "marmousi_eps_name": "m20_eps.pt",
        "read_weight_dir": "./checkpoints/",
        "sampled_dir": "./sampled_imgs/"
    }
    # for_all_cfa_sampling(config)
    random_generate(config)
