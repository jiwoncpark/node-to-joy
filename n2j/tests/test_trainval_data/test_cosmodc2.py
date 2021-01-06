import os
import unittest
import numpy as np
import shutil
import pandas as pd
from n2j.trainval_data.cosmodc2 import CosmoDC2
from n2j.trainval_data import coord_utils as cu


class TestCosmoDC2(unittest.TestCase):
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
        cls.fov = 6.0  # arcmin
        cls.halo_ra = np.array([1.0, 2.0, 0.5])/60.0  # deg
        cls.halo_dec = np.array([1.0, 0.5, 2.0])/60.0  # deg
        cls.cosmodc2 = CosmoDC2(cls.out_dir, test=True)
        cls.cosmodc2.healpix = 10450

    def test_get_generator(self):
        """Test if generator returns the correct number of rows and columns

        """
        n_gals = int(1e2)
        chunksize = 7
        data = {'halo_mass': 10**(np.random.randn(n_gals) + 11.0),
                'stellar_mass': (10.0**10.0)*np.ones(n_gals),
                'is_central': [True]*n_gals,
                'ra_true': np.random.rand(n_gals),  # deg
                'dec_true': np.random.rand(n_gals),  # deg
                'redshift_true': [1.0]*n_gals,
                'convergence': [0.01]*n_gals,
                'shear1': [0.01]*n_gals,
                'shear2': [0.01]*n_gals}
        df = pd.DataFrame(data)
        df.to_csv(self.cosmodc2.test_data_path, index=None)
        gen = self.cosmodc2.get_generator(['halo_mass', 'is_central'],
                                          chunksize=chunksize)
        # Test for loop
        n_rows = 0
        for df in gen:
            n_rows += df.shape[0]
        cols = df.columns.values
        cols.sort()
        np.testing.assert_equal(n_rows, n_gals)
        np.testing.assert_equal(cols, ['halo_mass', 'is_central'])

    def test_get_sightlines_on_grid(self):
        """Test number of sightlines and correctness of matching distance

        """
        ra, dec = cu.get_healpix_centers(self.cosmodc2.healpix,
                                         self.cosmodc2.nside,
                                         nest=False)
        n_gals = int(1e4)
        data = {'ra_true': np.random.rand(n_gals)*3.0 + (ra - 1.5),  # deg
                'dec_true': np.random.rand(n_gals)*3.0 + (dec - 1.5),  # deg
                'redshift_true': [2.1]*n_gals,
                'convergence': [0.01]*n_gals,
                'shear1': [0.01]*n_gals,
                'shear2': [0.01]*n_gals}
        df = pd.DataFrame(data)
        df.to_csv(self.cosmodc2.test_data_path, index=None)
        np.random.seed(123)
        N = 30
        resolution = 0.2  # deg
        sightlines = self.cosmodc2.get_sightlines_on_grid(N,
                                                          dist_thres=resolution)
        np.testing.assert_array_equal(sightlines.shape[0], N,
                                      err_msg='wrong number of sightlines')
        np.testing.assert_array_less(sightlines['eps'].values, resolution,
                                     err_msg='some LOS not satisfying dist cut')

    def test_get_los_halos_mass_dist_cut(self):
        """Test if the `get_los_halos` method applies the correct mass and
        position cuts

        """
        # Prepare input dataframe
        data = {'halo_mass': 10.0**np.array([10, 10.5, 11.05, 11.5, 12]),
                'stellar_mass': 10.0**np.array([10, 10, 10, 10, 10]),
                'is_central': [True]*5,
                'ra_true': np.array([1, 1, 1, 1, 2])/60.0,  # deg
                'dec_true': np.array([1, 1, 1, 1, 2])/60.0,  # deg
                'redshift_true': [1.0]*5}
        df = pd.DataFrame(data)
        df.to_csv(self.cosmodc2.test_data_path, index=None)
        # Generate halos
        halos = self.cosmodc2.get_los_halos(ra_los=0.0,
                                            dec_los=0.0,
                                            z_src=self.z_src,
                                            fov=self.fov,
                                            out_path=None)
        np.testing.assert_equal(halos.shape[0], 3)
        np.testing.assert_array_less(halos['z'].values, self.z_src)
        np.testing.assert_equal(np.all(halos['is_central'].values), True)
        np.testing.assert_array_less(11.0, np.log10(halos['halo_mass'].values))
        np.testing.assert_array_less(halos['dist'].values, self.fov)

    def test_get_los_halos_redshift_cut(self):
        """Test if the `get_los_halos` method applies the correct redshift and
        position cuts

        """
        # Prepare input dataframe
        data = {'halo_mass': 10.0**np.array([10, 10.5, 11.1, 11.5, 12, 12, 12]),
                'stellar_mass': 10.0**np.array([10, 10, 10, 10, 10, 10, 10]),
                'is_central': [False, True, True, True, True, False, True],
                'ra_true': np.array([1, 1, 1, 1, 2, 1, 1])/60.0,  # deg
                'dec_true': np.array([1, 1, 1, 1, 2, 1, 1])/60.0,  # deg
                'redshift_true': [1, 1, 1, 1, 2.5, 1, 1]}
        df = pd.DataFrame(data)
        df.to_csv(self.cosmodc2.test_data_path, index=None)
        # Generate halos
        halos = self.cosmodc2.get_los_halos(ra_los=0.0,
                                            dec_los=0.0,
                                            z_src=self.z_src,
                                            fov=self.fov,
                                            out_path=None)
        np.testing.assert_equal(halos.shape[0], 3)
        np.testing.assert_array_less(halos['z'].values, self.z_src)
        np.testing.assert_equal(np.all(halos['is_central'].values), True)
        np.testing.assert_array_less(11.0, np.log10(halos['halo_mass'].values))
        np.testing.assert_array_less(halos['dist'].values, self.fov)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)


if __name__ == '__main__':
    unittest.main()
