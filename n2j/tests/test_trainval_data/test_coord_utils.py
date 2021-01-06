import unittest
import numpy as np
import healpy as hp
from n2j.trainval_data import coord_utils as cu
from scipy import stats


class TestCoordUtils(unittest.TestCase):
    """A suite of tests verifying the raytracing utility methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        pass

    def test_get_target_nside(self):
        """Test if the correct target NSIDE is returned

        """
        # Say we want 17 subsamples of a healpix, close to 2 order diff (16)
        # Then we need to choose 3 order diff to sample more than 17
        order_diff = 2
        n_samples = int(4**order_diff + 1)
        order_in = 5
        nside_desired = int(2**(order_in + order_diff + 1))
        nside_actual = cu.get_target_nside(n_samples, nside_in=2**order_in)
        np.testing.assert_equal(nside_actual, nside_desired)

    def test_get_skycoord(self):
        """Test if a SkyCoord instance is returned

        """
        from astropy.coordinates import SkyCoord
        skycoord_actual = cu.get_skycoord(ra=np.array([0, 1, 2]),
                                          dec=np.array([0, 1, 2]))
        assert isinstance(skycoord_actual, SkyCoord)
        assert skycoord_actual.shape[0] == 3

    def test_sample_in_aperture(self):
        """Test uniform distribution of samples

        """
        radius = 3.0/60.0  # deg
        x, y = cu.sample_in_aperture(10000, radius=radius)
        r2 = x**2 + y**2
        ang = np.arctan2(y, x)
        uniform_rv_r2 = stats.uniform(loc=0, scale=radius**2.0)
        D, p = stats.kstest(r2, uniform_rv_r2.cdf)
        np.testing.assert_array_less(0.01, p, err_msg='R2 fails KS test')
        uniform_rv_ang = stats.uniform(loc=-np.pi, scale=2*np.pi)
        D, p = stats.kstest(ang, uniform_rv_ang.cdf)
        np.testing.assert_array_less(0.01, p, err_msg='angle fails KS test')

    def test_get_healpix_centers(self):
        """Test if correct sky locations are returned in the cosmoDC2 convention

        """
        # Correct answers hardcoded with known cosmoDC2 catalog values
        # Input i_pix is in nested scheme
        ra, dec = cu.get_healpix_centers(hp.ring2nest(32, 10450), 32, nest=True)
        np.testing.assert_array_almost_equal(ra, [67.5], decimal=1)
        np.testing.assert_array_almost_equal(dec, [-45.0], decimal=1)
        # Input i_pix is in ring scheme
        ra, dec = cu.get_healpix_centers(10450, 32, nest=False)
        np.testing.assert_array_almost_equal(ra, [67.5], decimal=1)
        np.testing.assert_array_almost_equal(dec, [-45.0], decimal=1)

    def test_upgrade_healpix(self):
        """Test correctness of healpix upgrading

        """
        nside_in = 2
        nside_out = nside_in*2  # must differ by 1 order for this test
        npix_in = hp.nside2npix(nside_in)
        npix_out = hp.nside2npix(nside_out)
        pix_i = 5
        # Upgrade pix_i in NSIDE=1 using cu
        # Downgrade all pixels in NSIDE=2 to NSIDE=1
        # Check if mappings from NSIDE=1 to NSIDE=2 match
        # Output is always NESTED
        # Test 1: Input pix_i is in NESTED
        # "visual" checks with https://healpix.jpl.nasa.gov/html/intronode4.htm
        actual = cu.upgrade_healpix(pix_i, True, nside_in, nside_out)
        desired_all = np.arange(npix_out).reshape((npix_in, 4))
        desired = np.sort(desired_all[pix_i, :]) # NESTED
        np.testing.assert_array_equal(desired, [20, 21, 22, 23], "visual")
        np.testing.assert_array_equal(actual, desired, "input in NESTED")
        # Test 2: Input pix_i is in RING
        actual = cu.upgrade_healpix(pix_i, False, nside_in, nside_out)
        # See https://stackoverflow.com/a/56675901
        # `reorder` reorders RING IDs in NESTED order
        # `reshape` is possible because the ordering is NESTED
        # indexing should be done with a NESTED ID because ordering is NESTED
        # but the output is in RING ID, which was reordered in the first place
        desired_all = hp.reorder(np.arange(npix_out), r2n=True).reshape((npix_in, 4))
        desired_ring = desired_all[hp.ring2nest(nside_in, pix_i), :]
        np.testing.assert_array_equal(np.sort(desired_ring),
                                      [14, 26, 27, 43],
                                      "visual")
        desired_nest = hp.ring2nest(nside_out, desired_ring)
        np.testing.assert_array_equal(np.sort(actual),
                                      np.sort(desired_nest),
                                      "input in RING")

    def test_match(self):
        """Test correctness of matching

        """
        ra_grid = np.array([1, 2, 3])
        dec_grid = np.array([1, 2, 3])
        ra_cat = np.array([1.1, 10, 20, 1.9, 30])
        dec_cat = np.array([1.1, 20, 10, 1.9, 30])
        fake_dist = np.sqrt(2*0.1**2.0)
        constraint, i_cat, dist = cu.match(ra_grid, dec_grid,
                                           ra_cat, dec_cat, 0.5)
        np.testing.assert_array_equal(constraint, [True, True, False])
        np.testing.assert_array_equal(i_cat, [0, 3])
        np.testing.assert_array_almost_equal(dist,
                                             [fake_dist, fake_dist],
                                             decimal=4)

    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()