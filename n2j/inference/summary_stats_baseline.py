"""Summary stats baseline computations

"""
import os
import os.path as osp
import copy
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
import torch
from torch_scatter import scatter_add
import n2j.inference.infer_utils as iutils


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
                 train_y, out_dir, test_y=None):
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
        self.test_y = test_y
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.overview_path = osp.join(self.out_dir, 'overview.csv')

    def match_summary_stats(self, thresholds, interim_pdf_func=None,
                            min_matches=1000):
        """Match summary stats between train and test

        Parameters
        ----------
        thresholds : dict
            Matching thresholds for summary stats
            Keys should be one or both of 'N' and 'N_inv_dist'.
        interim_pdf_func : callable, optional
            Interim prior PDF with which to reweight the samples

        """
        ss_names = list(thresholds.keys())
        n_test = len(self.test_stats.stats[ss_names[0]])
        overview = pd.DataFrame(columns=['los_i',
                                         'summary_stats_name',
                                         'threshold',
                                         'n_matches',
                                         'med',
                                         'plus_1sig',
                                         'minus_1sig',
                                         'logp'
                                         'mad',
                                         'mae',
                                         'is_optimal'])
        for i in tqdm(range(n_test), desc="matching"):
            for s in ss_names:
                test_x = self.test_stats.stats[s][i]
                optimal_crit = np.empty(len(thresholds[s]))
                rows_for_s = []
                for t_idx, t in enumerate(thresholds[s]):
                    # TODO: do this in chunks
                    accepted, _ = match(self.train_stats.stats[s],
                                        test_x,
                                        self.train_y,
                                        t)
                    n_matches = len(accepted)
                    np.save(osp.join(self.out_dir,
                                     f'matched_k_los_{i}_ss_{s}_{t:.0f}.npy'),
                            accepted)
                    # Add descriptive stats to overview table
                    row = dict(los_i=i,
                               summary_stats_name=s,
                               threshold=t,
                               test_x=test_x,
                               n_matches=n_matches)
                    optimal_crit[t_idx] = n_matches
                    if len(accepted) > 0:
                        if interim_pdf_func is not None:
                            inv_prior = 1.0/interim_pdf_func(accepted)
                            try:
                                resamples = iutils.resample_from_samples(accepted,
                                                                         inv_prior,
                                                                         n_resamples=10000,
                                                                         plot_path=None)
                            except ValueError:
                                print("Accepted samples were of shape", accepted.shape)
                            resamples = resamples.squeeze()  # [n_resamples]
                            np.save(osp.join(self.out_dir,
                                             f'matched_resampled_los_{i}_ss_{s}_{t:.0f}.npy'),
                                    resamples)
                        else:
                            resamples = accepted  # do not weight
                        lower, med, upper = np.quantile(resamples,
                                                        [0.5-0.34, 0.5, 0.5+0.34])
                        row.update(med=med,
                                   plus_1sig=upper-med,
                                   minus_1sig=med-lower,
                                   mad=stats.median_abs_deviation(resamples)
                                   )
                        # Comparison with truth, if available
                        if self.test_y is not None:
                            kde = stats.gaussian_kde(resamples,
                                                     bw_method='scott')
                            true_k = self.test_y[i, 0]
                            row.update(logp=kde.logpdf(true_k).item(),
                                       mae=np.median(np.abs(resamples - true_k)),
                                       true_k=true_k
                                       )
                    # Each ss name and threshold combo gets a row
                    # Wait until all thresholds are collected to append
                    rows_for_s.append(row)
                # Determine optimal threshold
                try:
                    is_optimal = get_optimal_threshold(thresholds[s],
                                                       optimal_crit,
                                                       min_matches=min_matches)
                except:
                    print("Summary stat: ", s)
                    print("Thresholds: ", thresholds[s])
                    print("Matches: ", optimal_crit)
                    raise ValueError("Can't find the optimal threshold!")
                # Record whether each row was "optimal"
                # There's only one optimal row for a given ss_name
                for r_i, r in enumerate(rows_for_s):
                    r.update(is_optimal=is_optimal[r_i])
                overview = overview.append(rows_for_s, ignore_index=True)
        print(f"Saving overview table at {self.overview_path}...")
        overview.to_csv(self.overview_path, index=False)

    def get_samples(self, idx, ss_name, threshold=None):
        """Get the pre-weighting (raw) accepted samples

        Parameters
        ----------
        idx : int
            ID of sightline
        ss_name : str
            Summary stats name
        threshold : int, optional
            Matching threshold. If None, use the optimal threshold.
            Default: None

        Returns
        -------
        np.ndarray
            Samples of shape `[n_matches]`
        """
        if threshold is None:
            # Default to optimal threshold
            overview = self.get_overview_table()
            crit = np.logical_and(np.logical_and(overview['los_i'] == idx,
                                  overview['summary_stats_name'] == ss_name),
                                  overview['is_optimal'])
            threshold = overview[crit]['threshold'].item()
        path = osp.join(self.out_dir,
                        f'matched_k_los_{idx}_ss_{ss_name}_{threshold:.0f}.npy')
        samples = np.load(path)
        return samples

    def get_overview_table(self):
        if not osp.exists(self.overview_path):
            raise OSError("Table doesn't exist. Please generate it first.")
        else:
            overview = pd.read_csv(self.overview_path, index_col=None)
        return overview


def get_optimal_threshold(thresholds, n_matches, min_matches=1000):
    """Get the smallest threshold that has some minimum number of matches

    Parameters
    ----------
    thresholds : array-like
    n_matches : array-like
    min_matches : int
    """
    is_optimal = np.zeros(len(thresholds)).astype(bool)  # init all False
    thresholds = np.array(thresholds).astype(float)
    n_matches = np.array(n_matches)
    # Impossible for thresholds with n_matches < min_matches to be selected
    thresholds[n_matches < min_matches] = np.nan  # hacky
    i = np.nanargmin(thresholds)
    is_optimal[i] = True
    return is_optimal


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
    accepted = train_y[is_passing]  # [n_test, Y_dim]
    return accepted.squeeze(-1), is_passing
