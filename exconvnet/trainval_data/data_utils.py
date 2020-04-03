"""Some utilities for loading data
"""

import numpy as np
import torch

__all__ = ['collate_fns']

collate_fns = {
    'RNN': collate_RNN
}

#######################################
# Declare all collate functions below #
#######################################

def collate_RNN(batch):
    """Collate function for the DataLoader when
    feeding to an RNN.

    Parameters
    ----------
    batch : array-like
        A batch where each element is a tuple (x, y), the return
        value for __getitem__ for the custom Dataset
    """
    pass


if __name__ == '__main__':
    pass
