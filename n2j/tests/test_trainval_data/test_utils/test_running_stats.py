"""Tests for computing mean and std in running batches

"""

import unittest
import torch
import numpy as np
from n2j.trainval_data.utils.running_stats import RunningStats


class TestRunningStats(unittest.TestCase):
    """A suite of tests verifying mean and std in running batches

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        np.random.seed(123)
        cls.n_data = 100000
        # Simulate random data
        cls.some_array = np.random.randn(cls.n_data, 2)*np.array([[0.5, 0.1]])
        #cls.some_array += np.random.randn(cls.n_data, 2)*np.array([[0.2, 0.0]])
        cls.some_array += np.array([[0.1, 0.3]])
        cls.some_array = np.exp(cls.some_array)  # make lognormal
        cls.mu_emp = cls.some_array.mean(axis=0, keepdims=True)
        cls.sig_emp = cls.some_array.std(axis=0, keepdims=True)
        cls.batch_size = 100

    def test_running_mean_np(self):
        """Test running mean computation on np array

        """
        running_mean = np.zeros([1, 2])
        for b in range(self.n_data//self.batch_size):
            new = self.some_array[b*self.batch_size:(b+1)*self.batch_size, :]
            running_mean += (new.mean(axis=0, keepdims=True) - running_mean)/(b+1)
        np.testing.assert_array_almost_equal(running_mean, self.mu_emp, decimal=5)

    def test_running_std_np(self):
        """Test running std computation on np array

        """
        running_mean = np.zeros([1, 2])
        running_var = np.zeros([1, 2])
        for b in range(self.n_data//self.batch_size):
            new = self.some_array[b*self.batch_size:(b+1)*self.batch_size, :]
            new_mean = new.mean(axis=0, keepdims=True)
            new_var = new.var(axis=0, keepdims=True)
            running_var += (new_var - running_var)/(b+1) + (b/(b+1)**2.0)*(running_mean - new_mean)**2.0
            running_mean += (new_mean - running_mean)/(b+1)
            # running_std += (new.std(axis=0, keepdims=True) - running_std)/(b+1)
        np.testing.assert_array_almost_equal(running_var**0.5, self.sig_emp, decimal=5)
        np.testing.assert_array_almost_equal(running_mean, self.mu_emp, decimal=5)

    def test_running_stats(self):
        """Test running mean, std computation on torch using RunningStats

        """
        some_array_torch = torch.tensor(self.some_array)
        loader_dict = dict(data=lambda x: x)
        rs = RunningStats(loader_dict)
        for b in range(self.n_data//self.batch_size):
            new = some_array_torch[b*self.batch_size:(b+1)*self.batch_size, :]
            rs.update(new, b)
        np.testing.assert_array_almost_equal(rs.stats['data_var']**0.5, self.sig_emp, decimal=5)
        np.testing.assert_array_almost_equal(rs.stats['data_mean'], self.mu_emp, decimal=5)


if __name__ == '__main__':
    unittest.main()
