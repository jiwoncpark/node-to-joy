"""Summary stats baseline computations

"""
import os.path as osp
import numpy as np
from tqdm import tqdm
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
        # TODO: don't hold all elements in memory, append to file
        # in chunks
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

    def set_stats(self, stats_path):
        """Loads a previously stored stats

        Parameters
        ----------
        stats_path : str
            Path to the .npy file of the stats dictionary
        """
        stats = np.load(stats_path, allow_pickle=True).item()
        self.stats = stats

    def export_stats(self, stats_path):
        """Exports the stats attribute to disk as a npy file

        Parameters
        ----------
        stats_path : str
            Path to the .npy file of the stats dictionary
        """
        np.save(stats_path, self.stats, allow_pickle=True)


class Matcher:
    def __init__(self, train_stats, test_stats,
                 train_y, out_dir):
        """Matcher of summary statistics between two datasets, train
        and test

        Parameters
        ----------
        train_stats : SummaryStatistics instance
        test_stats : SummaryStatistics instance
        train_y : np.ndarray
        out_dir : str
            Output dir for matched data products
        """
        self.train_stats = train_stats
        self.test_stats = test_stats
        self.train_y = train_y
        self.out_dir = out_dir

    def match_summary_stats(self, thresholds):
        """Match summary stats between train and test

        Parameters
        ----------
        thresholds : dict
            Matching thresholds for summary stats
            Keys should be one or both of 'N' and 'N_inv_dist'.

        """
        ss_names = list(thresholds.keys())
        n_test = len(self.test_stats[ss_names[0]])
        for i in tqdm(range(n_test), desc="matching"):
            for s in ss_names:
                test_x = self.test_stats.stats[s][i]
                for t in thresholds[s]:
                    # TODO: do this in chunks
                    accepted, _ = match(self.train_stats.stats[s],
                                        test_x,
                                        self.train_y,
                                        t)
                    np.save(osp.join(self.out_dir,
                                     f'matched_k_los_{i}_ss_{s}_{t:.5f}.npy'),
                            accepted)


def match(train_x, test_x, train_y, threshold):
    """Match summary stats between train and test within given threshold

    Parameters
    ----------
    train_x : np.ndarray
        train summary stats
    test_x : float
        test summary stats
    train_y : np.ndarray
        train target values
    threshold : float
        closeness threshold matching is based on

    Returns
    -------
    tuple
        boolean mask of accepted samples for train_y and the accepted
        samples
    """
    is_passing = np.abs(train_x - test_x) < threshold
    accepted = train_y[is_passing]
    return accepted, is_passing
