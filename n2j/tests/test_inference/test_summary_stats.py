"""Tests for n2j.inference.summary_stats

"""

import os
import shutil
import unittest
import numpy as np
from addict import Dict
import torch
import n2j.inference.summary_stats_baseline as ssb


class TestSummaryStats(unittest.TestCase):
    """A suite of tests verifying n2j.inference.summary_stats_utils
    utility functions

    """
    @classmethod
    def setUpClass(cls):
        cls.stats_path = 'stats_path_testing.npy'

    def test_get_number_counts(self):
        """Test `get_number_counts`
        """
        x = torch.randn([10, 3])
        batch_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
        actual = ssb.get_number_counts(x, batch_indices)
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

        actual = ssb.get_inv_dist_number_counts(torch.from_numpy(x),
                                               torch.from_numpy(batch_indices),
                                               ra_dec_idx)
        dist = np.array([5, np.sqrt(2)*1.e-7 + 1.e-5, 29, 13, 17, 25])
        expected = np.array([sum(1.0/dist[:3]),
                             sum(1.0/dist[3:4]),
                             sum(1.0/dist[4:])])
        np.testing.assert_array_almost_equal(actual, expected,
                                             decimal=4)

    def test_update(self):
        """Test `update` method
        """
        ss_obj = ssb.SummaryStats(n_data=3)
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
        dist = np.array([5, np.sqrt(2)*1.e-7 + 1.e-5, 29, 13, 17, 25])
        expected_N_inv_dist = np.array([sum(1.0/dist[:3]),
                                        sum(1.0/dist[3:4]),
                                        sum(1.0/dist[4:])])
        np.testing.assert_array_equal(ss_obj.stats['N'],
                                      expected_N)
        np.testing.assert_array_almost_equal(ss_obj.stats['N_inv_dist'],
                                             expected_N_inv_dist,
                                             decimal=4)

    def test_set_stats_export_stats(self):
        """Test `set_stats` method
        """
        any_stats = dict(
                         N=np.arange(5),
                         N_inv_dist=np.arange(5)
                         )
        np.save(self.stats_path, any_stats, allow_pickle=True)
        ss_obj = ssb.SummaryStats(n_data=5)
        ss_obj.set_stats(self.stats_path)
        np.testing.assert_array_equal(ss_obj.stats['N'],
                                      np.arange(5))
        np.testing.assert_array_equal(ss_obj.stats['N_inv_dist'],
                                      np.arange(5))
        ss_obj.export_stats(self.stats_path)
        ss_obj.set_stats(self.stats_path)
        np.testing.assert_array_equal(ss_obj.stats['N'],
                                      np.arange(5))
        np.testing.assert_array_equal(ss_obj.stats['N_inv_dist'],
                                      np.arange(5))

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.stats_path)


if __name__ == '__main__':
    unittest.main()
