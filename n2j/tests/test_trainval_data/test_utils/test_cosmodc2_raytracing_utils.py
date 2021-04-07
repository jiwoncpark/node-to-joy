"""Tests for the cosmodc2_raytracing_utils module

"""

import os
import unittest
import numpy as np
import shutil
import pandas as pd
from n2j.trainval_data import cosmodc2_raytracing_utils as ru
from n2j.trainval_data import halo_utils as hu


class TestCosmoDC2RaytracingUtils(unittest.TestCase):
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

    def test_get_los_halos_mass_dist_cut(self):
        """Test if the `get_los_halos` method applies the correct mass and
        position cuts

        """
        # Prepare input dataframe generator
        test_cosmodc2_path = os.path.join(self.out_dir, 'cosmodc2.csv')
        halo_cols = ['halo_mass', 'stellar_mass', 'is_central']
        halo_cols += ['ra_true', 'dec_true', 'redshift_true']
        df = pd.DataFrame({
                         'halo_mass': 10.0**np.array([10, 10.5, 11.1, 11.5, 12]),
                         'stellar_mass': 10.0**np.array([10, 10, 10, 10, 10]),
                         'is_central': [True]*5,
                         'ra_true': np.array([1, 1, 1, 1, 2])/60.0,  # deg
                         'dec_true': np.array([1, 1, 1, 1, 2])/60.0,  # deg
                         'redshift_true': [1.0]*5})
        df.to_csv(test_cosmodc2_path, index=None)
        df_gen = pd.read_csv(test_cosmodc2_path, index_col=None, chunksize=2)
        # Generate halos
        test_halos_path = os.path.join(self.out_dir, 'halos.csv')
        halos = ru.get_los_halos(df_gen,
                                 ra_los=0.0, dec_los=0.0,
                                 z_src=self.z_src,
                                 fov=6.0,
                                 mass_cut=11.0,
                                 out_path=test_halos_path)
        assert halos.shape[0] == 3

    def test_get_sightlines_on_grid(self):
        """Test number of sightlines and correctness of matching distance

        """
        np.random.seed(123)
        N = 30
        resolution = 0.2
        out_path = os.path.join(self.out_dir, 'sightlines.csv')
        sightlines = ru.get_sightlines_on_grid(self.healpix, N, out_path,
                                               dist_thres=resolution, test=True)
        np.testing.assert_array_equal(sightlines.shape[0], N,
                                      err_msg='wrong number of sightlines')
        np.testing.assert_array_less(sightlines['eps'].values, resolution,
                                     err_msg='some LOS not satisfying dist cut')

    def test_get_concentration(self):
        """Test mass-concentration relation at extreme values

        """
        c_0 = 3.19
        c200_at_stellar_mass = hu.get_concentration(1.0, 1.0,
                      m=-0.10, A=3.44, trans_M_ratio=430.49, c_0=c_0)
        c200_at_high_halo_mass = hu.get_concentration(10.0**5, 1.0,
                      m=-0.10, A=3.44, trans_M_ratio=430.49, c_0=c_0)
        np.testing.assert_almost_equal(c200_at_stellar_mass, 6.060380052400085,
                                       err_msg='halo mass at stellar mass')
        np.testing.assert_almost_equal(c200_at_high_halo_mass, c_0, decimal=2,
                                       err_msg='at high halo mass')

    def test_raytrace_single_sightline(self):
        """Test if the raytrace_single_sightline runs without error and outputs
        reasonable sightline kappa and resampled kappas

        """
        idx = 0
        ra_los = 0.0
        dec_los = 0.0
        z_src = 2.0
        fov = 6.0 # arcmin
        n_kappa_samples = 20
        mass_cut = 11.0
        halo_path = '{:s}/los_halos_sightline={:d}.csv'.format(self.out_dir, idx)
        halos = pd.DataFrame({
                             'halo_mass': self.halo_mass,
                             'stellar_mass': self.stellar_mass,
                             'z': self.halo_z,
                             'center_x': self.halo_ra*3600.0,
                             'center_y': self.halo_dec*3600.0,
                             })
        Rs, alpha_Rs, lensing_eff = hu.get_nfw_kwargs(halos['halo_mass'],
                                                      halos['stellar_mass'],
                                                      halos['z'],
                                                      self.z_src)
        halos['Rs'] = Rs
        halos['alpha_Rs'] = alpha_Rs
        halos['lensing_eff'] = lensing_eff
        halos.reset_index(drop=True, inplace=True)
        halos.to_csv(halo_path, index=None)
        ru.raytrace_single_sightline(idx, self.healpix, ra_los, dec_los,
                                     z_src, fov, False, False,
                                     n_kappa_samples, mass_cut,
                                     self.out_dir, test=True)
        kappa_samples_path = '{:s}/k_samples_sightline={:d}.npy'.format(self.out_dir, idx)
        kappa_samples = np.load(kappa_samples_path)
        assert len(kappa_samples) == n_kappa_samples
        np.testing.assert_array_less(kappa_samples, 1.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)


if __name__ == '__main__':
    unittest.main()