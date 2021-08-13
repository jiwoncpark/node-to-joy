"""Summary stats baseline computations

"""
import os
import numpy as np
import torch
from torch_scatter import scatter_add


def get_number_counts(x, batch_indices):
    """Get the unweighted number counts

    Parameters
    ----------
    x : torch.tensor
        Input features of shape [n_nodes, n_features] for a given batch
    batch_indices : torch.tensor
        Batch indices of shape [n_nodes,] for a given batch
    """
    ones = torch.ones(x.shape[0])
    N = scatter_add(ones, batch_indices)
    return N.numpy()


def get_inv_dist_number_counts(x, batch_indices, pos_indices):
    """Get the inverse-dist weighted number counts

    Parameters
    ----------
    x : torch.tensor
        Input features of shape [n_nodes, n_features] for a given batch
    batch_indices : torch.tensor
        Batch indices of shape [n_nodes,] for a given batch
    pos_indices : list
        List of the two indices corresponding to ra, dec in x

    """
    inv_dist = torch.sum(x[:, pos_indices]**2.0, dim=1)**0.5  # [n_nodes,]
    weights = 1.0/torch.maximum(inv_dist, torch.ones_like(inv_dist)*1.e-5)
    weighted_N = scatter_add(weights, batch_indices)
    return weighted_N.numpy()


class SummaryStats:
    def __init__(self, n_data, pos_indices=[0, 1]):
        """Summary stats calculator

        Parameters
        ----------
        n_data : int
            Size of dataset
        pos_indices : list
            Indices of `sub_features` corresponding to positions.
            By default, assumed to be the first two indices.

        """
        self.pos_indices = pos_indices
        # Init stats
        stats = dict()
        stats['N_inv_dist'] = np.zeros(n_data)
        stats['N'] = np.zeros(n_data)
        self.stats = stats

    def update(self, batch, i):
        """Update `stats` for a new batch

        Parameters
        ----------
        batch : array or dict
            new batch of data whose data can be accessed by the functions in
            `loader_dict`
        i : int
            index indicating that the batch is the i-th batch

        """
        x = batch.x
        batch_indices = batch.batch
        N = get_number_counts(x, batch_indices)
        N_inv_dist = get_inv_dist_number_counts(x, batch_indices,
                                                self.pos_indices)
        B = len(N)
        self.stats['N'][i*B: (i+1)*B] = N
        self.stats['N_inv_dist'][i*B: (i+1)*B] = N_inv_dist
