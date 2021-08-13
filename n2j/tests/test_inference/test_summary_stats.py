"""Tests for n2j.inference.summary_stats

"""

import os
import os.path as osp
import unittest
import shutil
import numpy as np
import pandas as pd
import scipy.stats
from addict import Dict
import torch
import n2j.inference.summary_stats as ss


class TestSummaryStats(unittest.TestCase):
    """A suite of tests verifying n2j.inference.summary_stats_utils
    utility functions

    """

    def test_get_number_counts(self):
        """Test `get_number_counts`
        """
        x = torch.randn([10, 3])
        batch_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
        actual = ss.get_number_counts(x, batch_indices)
        expected = [5, 2, 3]
        np.testing.assert_array_equal(actual, expected)

    def test_get_inv_dist_number_counts(self):
        """Test `get_inv_dist_number_counts`
        """
        x = np.ones([6, 3])
        batch_indices = np.array([0, 0, 0, 1, 2, 2])
        ra_dec_idx = [0, 1]
        x[:, ra_dec_idx] = np.array([[3, 4],
                                     [1.e-7, 1.e-7],
                                     [20, 21],
                                     [5, 12],
                                     [8, 15],
                                     [7, 24]])

        actual = ss.get_inv_dist_number_counts(torch.from_numpy(x),
                                               torch.from_numpy(batch_indices),
                                               ra_dec_idx)
        dist = np.array([5, 1.e-5, 29, 13, 17, 25])
        expected = np.array([sum(1.0/dist[:3]),
                             sum(1.0/dist[3:4]),
                             sum(1.0/dist[4:])])
        np.testing.assert_array_equal(actual, expected)

    def test_update(self):
        """Test `update` method
        """
        ss_obj = ss.SummaryStats(n_data=3)
        x_np = np.array([[3, 4],
                        [1.e-7, 1.e-7],
                        [20, 21],
                        [5, 12],
                        [8, 15],
                        [7, 24]])
        batch_indices_np = np.array([0, 0, 0, 1, 2, 2])
        batches = []
        batches.append(Dict(
                            x=torch.tensor(x_np),
                            batch=torch.tensor(batch_indices_np)
                            ))
        for i, b in enumerate(batches):
            ss_obj.update(b, i)
        # Compute expected
        expected_N = [3, 1, 2]
        dist = np.array([5, 1.e-5, 29, 13, 17, 25])
        expected_N_inv_dist = np.array([sum(1.0/dist[:3]),
                                        sum(1.0/dist[3:4]),
                                        sum(1.0/dist[4:])])
        np.testing.assert_array_equal(ss_obj.stats['N'],
                                      expected_N)
        np.testing.assert_array_equal(ss_obj.stats['N_inv_dist'],
                                      expected_N_inv_dist)


if __name__ == '__main__':
    unittest.main()
