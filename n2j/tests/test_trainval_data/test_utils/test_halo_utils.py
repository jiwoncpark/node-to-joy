"""Tests for the halo_utils module

"""

import os
import unittest
import shutil
import numpy as np
from n2j.trainval_data.utils import halo_utils as hu


class TestHaloUtils(unittest.TestCase):
    """A suite of tests verifying the raytracing utility methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests
        """
        cls.healpix = 10450
        cls.out_dir = 'test_out_dir'
        os.makedirs(cls.out_dir, exist_ok=True)
        cls.halo_mass = np.array([1e12, 5e12, 1e13])
        cls.stellar_mass = np.array([1e12, 5e12, 1e13])
        cls.halo_z = np.array([1.5, 1.0, 1.2])
        cls.z_src = 2.0
        cls.halo_ra = np.array([1.0, 2.0, 0.5])/60.0  # deg
        cls.halo_dec = np.array([1.0, 0.5, 2.0])/60.0  # deg

    def test_get_concentration(self):
        """Test mass-concentration relation at extreme values

        """
        c_0 = 3.19
        c200_at_stellar_mass = hu.get_concentration(1.0, 1.0, m=-0.10,
                                                    A=3.44,
                                                    trans_M_ratio=430.49,
                                                    c_0=c_0,
                                                    add_noise=False)
        c200_at_high_halo_mass = hu.get_concentration(10.0**5, 1.0, m=-0.10,
                                                      A=3.44,
                                                      trans_M_ratio=430.49,
                                                      c_0=c_0,
                                                      add_noise=False)
        np.testing.assert_almost_equal(c200_at_stellar_mass, 6.060380052400085,
                                       err_msg='halo mass at stellar mass')
        np.testing.assert_almost_equal(c200_at_high_halo_mass, c_0, decimal=2,
                                       err_msg='at high halo mass')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)


if __name__ == '__main__':
    unittest.main()
