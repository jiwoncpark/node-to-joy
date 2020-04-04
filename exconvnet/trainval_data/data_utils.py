"""Some utilities for loading data.

In particular, this file mainly contains collation
functions for the pytorch data loader (as we may treat
each batch differently depending on the architecture used).
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

__all__ = ['collate_RNN']


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

    Returns
    -------
    X_tr : torch.Tensor
        training mini-batch to be fed directly to the RNN
    """

    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    x_lens = [len(x) for x,y in sorted_batch]
    x_padded = pad_sequence([x for x,y in sorted_batch], batch_first=True)

    # pack this
    X_tr = pack_padded_sequence(x_padded, x_lens, batch_first=True, enforce_sorted=True)

    Y_tr = torch.Tensor([y for x,y in batch])

    return X_tr, Y_tr

if __name__ == '__main__':
    pass
