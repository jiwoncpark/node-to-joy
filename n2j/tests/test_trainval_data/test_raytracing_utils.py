import unittest
import numpy as np
from n2j.trainval_data import raytracing_utils as ru


class TestRaytracingUtils(unittest.TestCase):
    """A suite of tests verifying the raytracing utility methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests
        """
        cls.halo_mass = np.array([1e12, 5e12, 1e13])
        cls.stellar_mass = np.array([1e12, 5e12, 1e13])
        cls.halo_z = np.array([1.5, 1.0, 1.2])
        cls.z_src = 2.0
        cls.halo_ra = np.array([1.0, 2.0, 0.5])/60.0  # deg
        cls.halo_dec = np.array([1.0, 0.5, 2.0])/60.0  # deg
        from astropy.cosmology import WMAP7
        cls.cosmo = WMAP7

    def test_get_nfw_kwargs(self):
        """Test shapes of output for nfw_kwargs

        """
        np.random.seed(123)
        Rs, alpha_Rs, lensing_eff = ru.get_nfw_kwargs(self.halo_mass,
                                                      self.stellar_mass,
                                                      self.halo_z,
                                                      self.z_src,
                                                      self.cosmo)
        n_halos = len(self.halo_mass)
        np.testing.assert_equal(Rs.shape, (n_halos,))
        np.testing.assert_equal(alpha_Rs.shape, (n_halos,))
        np.testing.assert_equal(lensing_eff.shape, (n_halos,))

    def test_get_concentration(self):
        """Test mass-concentration relation at extreme values

        """
        c_0 = 3.19
        c200_at_stellar_mass = ru.get_concentration(1.0, 1.0,
                                                    m=-0.10,
                                                    A=3.44,
                                                    trans_M_ratio=430.49,
                                                    c_0=c_0)
        c200_at_high_halo_mass = ru.get_concentration(10.0**5,
                                                      1.0,
                                                      m=-0.10,
                                                      A=3.44,
                                                      trans_M_ratio=430.49,
                                                      c_0=c_0)
        np.testing.assert_almost_equal(c200_at_stellar_mass, 6.060380052400085,
                                       err_msg='halo mass at stellar mass')
        np.testing.assert_almost_equal(c200_at_high_halo_mass, c_0, decimal=2,
                                       err_msg='at high halo mass')


if __name__ == '__main__':
    unittest.main()