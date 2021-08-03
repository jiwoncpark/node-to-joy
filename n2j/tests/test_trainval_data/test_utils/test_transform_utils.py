import unittest
import numpy as np
from n2j.trainval_data.utils.transform_utils import MagErrorSimulator


class TestMagErrorSimulator(unittest.TestCase):
    """A suite of tests verifying the magnitude MagErrorSimulator methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        cls.es = MagErrorSimulator()

    def test_basic(self):
        """Verify correspondence with same analytic function in Collab notebook

        """
        mag = 24
        sigma_u = self.es.calculate_photo_err('u', mag)
        np.testing.assert_almost_equal(sigma_u, 0.04455297880298702)
        # if using OpSim median sky brightness, rather than tuned sky brightness:
        # np.testing.assert_almost_equal(sigma_u, 0.04086201169946179)
        # if using Table 2 seeing, sky brightness, rather than OpSim:
        # np.testing.assert_equal(sigma_u, 0.033833253781615086)

        mags = np.array([[24, 24, 24, 24, 24, 24]])
        sigmas = self.es.get_sigmas(mags)
        expected_sigmas = np.array([[0.04455298, 0.01766673, 0.01868476, 0.03230199, 0.09370100, 0.18518317]])
        # if using OpSim median sky brightness and seeing, rather than tuned sky brightness and seeing:
        # expected_sigmas = np.array([[0.04086201, 0.01766672, 0.01868475, 0.03230198, 0.05806892, 0.12310554]])
        # if using Table 2 seeing, sky brightness, rather than OpSim:
        # expected_sigmas = np.array([[0.03383325, 0.01618119, 0.01696213, 0.02418282, 0.04190728, 0.09457142]])
        np.testing.assert_array_almost_equal(sigmas, expected_sigmas, decimal=5)

    def test_r_band_lit(self):
        """Test if calculate_photo_err() r band values match Table 3 values
        for mags 21-24, single visit and 10 year depths

        """
        # arbitrary mags array for es instantiation, we don't use this
        es1 = MagErrorSimulator(depth=10)
        es2 = MagErrorSimulator(depth='single_visit')
        table_3_r_sigma = {'single_visit': [0.01, 0.02, 0.04, 0.10], 10: [0.005, 0.005, 0.006, 0.009]}

        for mag in range(21, 25):
            i = mag-21
            sigma_r = es1.calculate_photo_err('r', mag)
            np.testing.assert_almost_equal(sigma_r, table_3_r_sigma[10][i], decimal=1)

            sigma_r = es2.calculate_photo_err('r', mag)
            np.testing.assert_almost_equal(sigma_r, table_3_r_sigma['single_visit'][i], decimal=1)

    def test_sigma_rand_m_5(self):
        """Test sigma_rand = 0.2 for mag = m_5 (definition of 5-sigma depth)

        """

        all_bands = 'ugrizy'

        for band in all_bands:
            i = all_bands.find(band)
            m_5 = self.es.calculate_5sigma_depth_from_scratch(i)
            sigma_rand = self.es.calculate_rand_err(band, m_5) ** 0.5

            assert (sigma_rand == 0.2)

    def test_sigma_rand_infinite_brightness(self):
        """Test that sigma_rand = 0 for mag = -inf

        """
        all_bands = 'ugrizy'

        for band in all_bands:
            m = -np.inf
            sigma_rand = self.es.calculate_rand_err(band, m) ** 0.5
            #print(sigma_rand)
            assert (sigma_rand == 0)

    def test_shapes(self):
        in_mags = 22.0 + np.random.normal(size=[15, 8])
        out_mags = self.es(in_mags)
        np.testing.assert_equal(out_mags.shape, [15, 8])

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
