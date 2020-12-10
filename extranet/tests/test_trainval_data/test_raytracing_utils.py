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
        bounds = ray_util.get_healpix_bounds(self.healpix)
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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)

if __name__ == '__main__':
    unittest.main()