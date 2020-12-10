import os
import unittest
import numpy as np
import shutil
import pandas as pd
from extranet.trainval_data import raytracing_utils as ray_util

class TestRaytracingUtils(unittest.TestCase):
    """A suite of tests verifying the raytracing utility methods
    
    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests
        """
        cls.healpix = 10450
        cls.out_dir = 'test_out_dir'
        os.makedirs(cls.out_dir, exist_ok=True)

    def test_get_healpix_bounds(self):
        bounds = ray_util.get_healpix_bounds(self.healpix, edge_buffer=3.0/60.0)
        assert bounds['min_ra'] < bounds['max_ra']
        assert bounds['min_dec'] < bounds['max_dec']

    def test_get_prestored_healpix_bounds(self):
        bounds = ray_util.get_prestored_healpix_bounds(self.healpix)
        assert bounds['min_ra'] < bounds['max_ra']
        assert bounds['min_dec'] < bounds['max_dec']

    def test_get_sightlines_random(self):
        # Test number of sightlines
        for N in [1, 1001]:
            out_path = os.path.join(self.out_dir, 'random_sightlines.csv')
            ray_util.get_sightlines_random(self.healpix, N, out_path)
            df = pd.read_csv(out_path, index_col=None)
            assert df.shape[0] == N

    def test_get_nfw_kwargs(self):
        """Test if the vectorization across halos returns the same values
        as an explicit loop

        """
        # Halo info for one sightline
        halo_mass = np.array([1e12, 5e12, 1e13])
        stellar_mass = np.array([1e12, 5e12, 1e13])
        halo_z = np.array([1.5, 1.0, 1.2])
        z_src = 2.0
        def get_nfw_kwargs_loop(halo_mass, stellar_mass, halo_z, z_src):
            from lenstronomy.Cosmo.lens_cosmo import LensCosmo
            from astropy.cosmology import WMAP7   # WMAP 7-year cosmology
            c_200 = ray_util.get_concentration(halo_mass, stellar_mass)
            n_halos = len(halo_mass)
            halo_Rs, halo_alpha_Rs = np.empty(n_halos), np.empty(n_halos)
            for halo_i in range(n_halos):
                lens_cosmo = LensCosmo(z_lens=halo_z[halo_i], z_source=z_src, cosmo=WMAP7)
                Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=halo_mass[halo_i],
                                                                   c=c_200[halo_i])
                rho0, Rs, c, r200, M200 = lens_cosmo.nfw_angle2physical(Rs_angle=Rs_angle, 
                                                                        alpha_Rs=alpha_Rs)
                halo_Rs[halo_i] = Rs
                halo_alpha_Rs[halo_i] = alpha_Rs
            return halo_Rs, halo_alpha_Rs
        np.random.seed(123)
        Rs, alpha_Rs = get_nfw_kwargs_loop(halo_mass, stellar_mass, halo_z, z_src)
        np.random.seed(123)
        Rs_vec, alpha_Rs_vec = ray_util.get_nfw_kwargs(halo_mass, stellar_mass, halo_z, z_src)
        np.testing.assert_array_almost_equal(Rs_vec, Rs)
        np.testing.assert_array_almost_equal(alpha_Rs_vec, alpha_Rs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)

if __name__ == '__main__':
    unittest.main()