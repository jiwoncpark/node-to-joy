"""Utility functions for processing data used for training and validation

"""


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean)/self.std


class Slicer:
    def __init__(self, feature_idx):
        self.feature_idx = feature_idx

    def __call__(self, x):
        return x[:, self.feature_idx]