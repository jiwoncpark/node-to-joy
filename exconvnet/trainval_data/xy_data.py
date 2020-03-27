import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ExConvDataset(Dataset):
    """Represents the x/y data used to train or validate our models

    """
    def __init__(self, x_path, y_path, data_cfg):
        """
        Parameters
        ----------
        x_path : str
            path to the .npy file containing X
        y_path : str
            path to the .npy file containing Y
        """
        # get X and Y loaded
        self.X = np.load(x_path, allow_pickle=True)
        self.Y = np.load(y_path, allow_pickle=True)

        self.n_data = self.X.shape[0]
    def __len__(self):
        return self.n_data
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.Y[idx])

class ExConvTestSet(dataset):
    """Represents the XData used to test our models

    """
    def __init__(self, x_path):
        """
        Parameters
        ----------
        x_path : str
            path to the .npy file containing X
        """
        self.X = np.load(x_path, allow_pickle=True)

        self.n_data = self.X.shape[0]
    def __len__(self):
        return self.n_data
    def __getitem__(self, idx):
        return self.X[idx]
