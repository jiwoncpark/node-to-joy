"""Tests for the n2j.inference.infer_utils utility functions

"""

import os
import unittest
import shutil
import numpy as np
import pandas as pd
import scipy.stats
import n2j.inference.infer_utils as iutils
import n2j.data as in_data


class TestInferUtils(unittest.TestCase):
    """A suite of tests verifying n2j.inference.infer_utils utility functions

    """

    @classmethod
    def setUpClass(cls):
        """Set up seeding

        """
        cls.rng = np.random.default_rng(123)

    def test_get_normal_logpdf(self):
        mu = 0.04
        log_sigma = np.log(0.005)
        x = self.rng.normal(100)*np.exp(log_sigma) + 0.04
        actual_logp = iutils.get_normal_logpdf(mu, log_sigma, x)
        actual_logp = actual_logp - 0.5*np.log(2*np.pi)
        true_logp = scipy.stats.norm.logpdf(x,
                                            loc=mu, scale=np.exp(log_sigma))
        np.testing.assert_array_almost_equal(actual_logp, true_logp)

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
