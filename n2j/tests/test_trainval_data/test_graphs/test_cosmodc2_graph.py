"""Tests for the n2j.trainval_data.graphs.cosmodc2_graph.CosmoDC2Graph class

"""

import os
import unittest
import shutil
import numpy as np
import pandas as pd
from n2j.trainval_data.graphs.cosmodc2_graph import CosmoDC2Graph
from n2j import data
import n2j.trainval_data.coord_utils as cu


def get_true_dist(ra_gals, dec_gals):
    """Get the distance between some galaxies and some zeropoint sightline

    Parameters
    ----------
    ra_gals : `np.array`
    dec_gals : `np.array`

    Returns
    -------
    `np.array`
        distance in arcmin

    """
    dist, ra_diff, dec_diff = cu.get_distance(ra_gals, dec_gals, 0.0, 0.0)
    return dist*60.0


class TestCosmoDC2Graph(unittest.TestCase):
    """A suite of tests verifying CosmoDC2Graph class methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests
        """
        cls.healpix = 10450
        # Create labels for fake sightlines
        cls.raytracing_out_dir = 'raytracing_debug'
        os.makedirs(cls.raytracing_out_dir, exist_ok=True)
        sightlines = pd.DataFrame({'galaxy_id': [0, 1, 2],
                                   'ra': [0, 0, 0],
                                   'dec': [0, 0, 0],
                                   'final_kappa': [0.1, 0.2, -0.1],
                                   'final_gamma1': [0.1, 0.1, 0.1],
                                   'final_gamma2': [0.05, -0.05, 0.0]})
        sightlines.to_csv(os.path.join(cls.raytracing_out_dir,
                                       'sightlines.csv'), index=None)
        # Create fake photometric catalog for LOS galaxy properties
        cls.photometric = pd.DataFrame({'ra_true': [1/60.0, 1/60.0, 3],  # deg
                                        'dec_true': [1/60.0, 0.75/60.0, 3],
                                        'size': [0.5, 1.0, 2.0],
                                        'mag_i_lsst': [19, 20, 30]})
        photometric_path = os.path.join(data.__path__[0],
                                        'cosmodc2_{:d}'.format(cls.healpix),
                                        'raw',
                                        'debug_gals.csv')
        cls.photometric.to_csv(photometric_path, index=None)
        # Instantiate Dataset
        features = ['ra_true', 'dec_true', 'size', 'mag_i_lsst']
        cls.dataset = CosmoDC2Graph(healpix=cls.healpix,
                                    raytracing_out_dir=cls.raytracing_out_dir,
                                    aperture_size=3.0,
                                    n_data=3,
                                    features=features,
                                    debug=True,)

    def test_get_edges(self):
        """Test edge building

        """
        # If all rows of self.photometric are neighbors
        ra_dec_gals = self.photometric[['ra_true', 'dec_true']]
        ra_dec = np.vstack([np.zeros(2), ra_dec_gals])
        edge_index_np = self.dataset.get_edges(ra_dec).cpu().numpy()
        edge_index_set = set(zip(edge_index_np[0], edge_index_np[1]))
        # All neighbors connect to node 0
        # There's a closeness edge between nodes 1, 2
        assert edge_index_set == {(0, 0), (1, 0), (2, 0), (3, 0), (1, 2)}

    def test_len(self):
        """Test length metadata

        """
        np.testing.assert_equal(len(self.dataset), self.dataset.n_data)

    def test_get(self):
        """Test indexing

        """
        single_example = self.dataset[0]
        # Test if x, y, edge_index are all in there
        np.testing.assert_equal(len(single_example), 3,
                                err_msg="length of each training example"
                                        "should be 3, from x, y, edge_index")
        # Test shapes
        keep_dist = get_true_dist(self.photometric['ra_true'].values,
                                  self.photometric['dec_true'].values)
        keep_mag = np.array([True, True, False])
        keep = np.logical_and(keep_dist, keep_mag)
        n_neighbors = sum(keep)
        np.testing.assert_equal(single_example['x'].shape,
                                [n_neighbors+1, self.dataset.n_features])
        np.testing.assert_equal(single_example['y'].shape,
                                [1, 3])
        np.testing.assert_equal(single_example['edge_index'].shape,
                                [2, n_neighbors+2])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.raytracing_out_dir)
        for f in cls.dataset.raw_paths:
            os.remove(f)
        for f in cls.dataset.processed_paths:
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
