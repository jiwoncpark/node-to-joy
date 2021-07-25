import unittest
import numpy as np
from n2j.trainval_data.utils.transform_utils import ErrorSimulator


class TestTransformUtils(unittest.TestCase):
    """A suite of tests verifying the magnitude ErrorSimulator methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        cls.mags = np.array([[24, 24, 24, 24, 24, 24]])
        cls.es = ErrorSimulator(cls.mags)

    def test_basic(self):
        """Verify correspondence with same analytic function in Collab notebook

        """
        mag = 24
        sigma_u = self.es.calculate_photo_err('u', mag)
        np.testing.assert_equal(sigma_u, 0.033833253781615086)

        sigmas = self.es.get_sigmas()
        expected_sigmas = np.array([[0.03383325, 0.01618119, 0.01696213, 0.02418282, 0.04190728, 0.09457142]])
        np.testing.assert_array_almost_equal(sigmas, expected_sigmas, decimal=7)

    def test_r_band_lit(self):
        """Test if calculate_photo_err() r band values match Table 3 values
        for mags 21-24, single visit and 10 year depths

        """
        # arbitrary mags array for es instantiation, we don't use this
        es1 = ErrorSimulator(self.mags, depth=10)
        es2 = ErrorSimulator(self.mags, depth='single_visit')
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

            assert (sigma_rand == 0)

    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()