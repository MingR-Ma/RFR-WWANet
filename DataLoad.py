import numpy as np
from torch.utils.data import Dataset
import torch


class UpperAbdomenDataset(Dataset):

    def __init__(self, data_list):
        """
        Numpy data for training RFRA-Net.
        :param data_list:
        :return:
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = np.load(img_path)
        img = img[None, ...]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        return img
