import unittest
import numpy as np
import torch
from n2j.trainval_data.utils.transform_utils import (MagErrorSimulator,
                                                     MagErrorSimulatorTorch,
                                                     get_bands_in_x)


def test_get_bands_in_x():
    # Watch the 'Y' being uppercase
    x_cols = ['froyo', 'madi', 'mag_i_lsst', 'jiwon', 'mag_Y_lsst']
    mag_idx, which_bands = get_bands_in_x(x_cols)
    np.testing.assert_equal(mag_idx, [2, 4])
    np.testing.assert_equal(which_bands, ['i', 'y'])


class TestMagErrorSimulator(unittest.TestCase):
    """A suite of tests verifying the magnitude MagErrorSimulator methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        pass

    def test_basic(self):
        """Verify correspondence with same analytic function in Collab notebook

        """
        mes = MagErrorSimulator()
        mag = 24
        u_idx = list('ugrizy').index('u')  # 0
        sigma_u = mes.get_sigmas(mag)[:, u_idx]
        np.testing.assert_almost_equal(sigma_u, 0.04455297880298702)
        # if using OpSim median sky brightness, rather than tuned sky brightness:
        # np.testing.assert_almost_equal(sigma_u, 0.04086201169946179)
        # if using Table 2 seeing, sky brightness, rather than OpSim:
        # np.testing.assert_equal(sigma_u, 0.033833253781615086)

        mags = np.array([[24, 24, 24, 24, 24, 24]])
        sigmas = mes.get_sigmas(mags)
        expected_sigmas = np.array([[0.04455298, 0.01766673, 0.01868476,
                                   0.03230199, 0.09370100, 0.18518317]])
        # if using OpSim median sky brightness and seeing, rather than tuned sky brightness and seeing:
        # expected_sigmas = np.array([[0.04086201, 0.01766672, 0.01868475, 0.03230198, 0.05806892, 0.12310554]])
        # if using Table 2 seeing, sky brightness, rather than OpSim:
        # expected_sigmas = np.array([[0.03383325, 0.01618119, 0.01696213, 0.02418282, 0.04190728, 0.09457142]])
        np.testing.assert_array_almost_equal(sigmas, expected_sigmas, decimal=5)

    def test_r_band_lit(self):
        """Test if calculate_photo_err() r band values match Table 3 values
        for mags 21-24, single visit and 10 year depths

        """
        # arbitrary mags array for mes instantiation, we don't use this
        mes1 = MagErrorSimulator(depth=10)
        mes2 = MagErrorSimulator(depth='single_visit')
        table_3_r_sigma = {'single_visit': [0.01, 0.02, 0.04, 0.10],
                           10: [0.005, 0.005, 0.006, 0.009]}
        r_idx = list('ugrizy').index('r')  # 2
        for i, mag in enumerate(range(21, 25)):
            sigma_r = mes1.get_sigmas(mag)[:, r_idx]
            np.testing.assert_almost_equal(sigma_r,
                                           table_3_r_sigma[10][i],
                                           decimal=1)

            sigma_r = mes2.get_sigmas(mag)[:, r_idx]
            np.testing.assert_almost_equal(sigma_r,
                                           table_3_r_sigma['single_visit'][i],
                                           decimal=1)

    def test_sigma_rand_m_5(self):
        """Test sigma_rand = 0.2 for mag = m_5 (definition of 5-sigma depth)

        """
        mes = MagErrorSimulator()
        m_5 = mes.calculate_5sigma_depths()
        sigma_rand = mes.calculate_rand_err(m_5)**0.5
        np.testing.assert_equal(sigma_rand, 0.2)

    def test_sigma_rand_infinite_brightness(self):
        """Test that sigma_rand = 0 for mag = -inf

        """
        mes = MagErrorSimulator()
        m = -np.inf
        sigma_rand = mes.calculate_rand_err(m)**0.5
        np.testing.assert_equal(sigma_rand, 0.0)

    def test_shapes_call(self):
        """Test output shapes of __call__ given input x

        """
        mes = MagErrorSimulator()
        # batched x
        in_x = 22.0 + np.random.normal(size=[15, 8])
        out_mags = mes(in_x)
        np.testing.assert_equal(out_mags.shape, [15, 8])

    def test_shapes_get_sigmas(self):
        """Test output shapes of get_sigmas given input mags

        """
        # All 6 bands, scalar mag
        mes = MagErrorSimulator(mag_idx=None,  # doesn't matter here
                                which_bands=list('ugrizy'))  # default
        in_mags = 22.0 + np.random.normal()
        out_sigmas = mes.get_sigmas(in_mags)
        np.testing.assert_equal(out_sigmas.shape, [1, 6])
        # Partial bands, scalar mag
        mes = MagErrorSimulator(mag_idx=None,  # doesn't matter here
                                which_bands=['z', 'i', 'g'])
        in_mags = 22.0 + np.random.normal()
        out_sigmas = mes.get_sigmas(in_mags)
        np.testing.assert_equal(out_sigmas.shape, [1, 3])
        # Partial bands, vector mag
        mes = MagErrorSimulator(mag_idx=None,  # doesn't matter here
                                which_bands=['z', 'i', 'g'])
        in_mags = 22.0 + np.random.normal(size=[15, 3])
        out_sigmas = mes.get_sigmas(in_mags)
        np.testing.assert_equal(out_sigmas.shape, [15, 3])

    @classmethod
    def tearDownClass(cls):
        pass


class TestMagErrorSimulatorTorch(unittest.TestCase):
    """A suite of tests comparing MagErrorSimulator with MagErrorSimulatorTorch

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        pass

    def test_basic(self):
        """Verify correspondence with same analytic function in Collab notebook

        """
        mes = MagErrorSimulatorTorch()
        mag = 24
        u_idx = list('ugrizy').index('u')  # 0
        sigma_u = mes.get_sigmas(mag)[:, u_idx]
        np.testing.assert_almost_equal(sigma_u, 0.04455297880298702)
        # if using OpSim median sky brightness, rather than tuned sky brightness:
        # np.testing.assert_almost_equal(sigma_u, 0.04086201169946179)
        # if using Table 2 seeing, sky brightness, rather than OpSim:
        # np.testing.assert_equal(sigma_u, 0.033833253781615086)

        mags = torch.tensor([[24, 24, 24, 24, 24, 24]])
        sigmas = mes.get_sigmas(mags)
        expected_sigmas = torch.tensor([[0.04455298, 0.01766673, 0.01868476,
                                       0.03230199, 0.09370100, 0.18518317]])
        # if using OpSim median sky brightness and seeing, rather than tuned sky brightness and seeing:
        # expected_sigmas = np.array([[0.04086201, 0.01766672, 0.01868475, 0.03230198, 0.05806892, 0.12310554]])
        # if using Table 2 seeing, sky brightness, rather than OpSim:
        # expected_sigmas = np.array([[0.03383325, 0.01618119, 0.01696213, 0.02418282, 0.04190728, 0.09457142]])
        np.testing.assert_array_almost_equal(sigmas, expected_sigmas, decimal=5)

    def test_r_band_lit(self):
        """Test if calculate_photo_err() r band values match Table 3 values
        for mags 21-24, single visit and 10 year depths

        """
        # arbitrary mags array for mes instantiation, we don't use this
        mes1 = MagErrorSimulatorTorch(depth=10)
        mes2 = MagErrorSimulatorTorch(depth='single_visit')
        table_3_r_sigma = {'single_visit': [0.01, 0.02, 0.04, 0.10],
                           10: [0.005, 0.005, 0.006, 0.009]}
        r_idx = list('ugrizy').index('r')  # 2
        for i, mag in enumerate(range(21, 25)):
            sigma_r = mes1.get_sigmas(mag)[:, r_idx]
            np.testing.assert_almost_equal(sigma_r,
                                           table_3_r_sigma[10][i],
                                           decimal=1)

            sigma_r = mes2.get_sigmas(mag)[:, r_idx]
            np.testing.assert_almost_equal(sigma_r,
                                           table_3_r_sigma['single_visit'][i],
                                           decimal=1)

    def test_sigma_rand_m_5(self):
        """Test sigma_rand = 0.2 for mag = m_5 (definition of 5-sigma depth)

        """
        mes = MagErrorSimulatorTorch()
        m_5 = mes.calculate_5sigma_depths()
        sigma_rand = mes.calculate_rand_err(m_5)**0.5
        np.testing.assert_array_equal(sigma_rand,
                                      torch.ones([1, 6])*0.2)

    def test_sigma_rand_infinite_brightness(self):
        """Test that sigma_rand = 0 for mag = -inf

        """
        mes = MagErrorSimulatorTorch()
        m = -np.inf
        sigma_rand = mes.calculate_rand_err(m)**0.5
        np.testing.assert_array_equal(sigma_rand,
                                      torch.zeros([1, 6]))

    def test_shapes_call(self):
        """Test output shapes of __call__ given input x

        """
        mes = MagErrorSimulatorTorch()
        # batched x
        in_x = 22.0 + torch.normal(mean=0, std=1, size=[15, 8])
        out_mags = mes(in_x)
        np.testing.assert_array_equal(out_mags.shape, [15, 8])

    def test_shapes_get_sigmas(self):
        """Test output shapes of get_sigmas given input mags

        """
        # All 6 bands, scalar mag
        mes = MagErrorSimulatorTorch(mag_idx=None,  # doesn't matter here
                                     which_bands=list('ugrizy'))  # default
        in_mags = 22.0
        out_sigmas = mes.get_sigmas(in_mags)
        np.testing.assert_array_equal(out_sigmas.shape, [1, 6])
        # Partial bands, scalar mag
        mes = MagErrorSimulatorTorch(mag_idx=None,  # doesn't matter here
                                     which_bands=['z', 'i', 'g'])
        in_mags = 22.0
        out_sigmas = mes.get_sigmas(in_mags)
        np.testing.assert_array_equal(out_sigmas.shape, [1, 3])
        # Partial bands, vector mag
        mes = MagErrorSimulatorTorch(mag_idx=None,  # doesn't matter here
                                     which_bands=['z', 'i', 'g'])
        in_mags = 22.0 + torch.normal(mean=torch.zeros([15, 3]),
                                      std=torch.ones([15, 3]))
        out_sigmas = mes.get_sigmas(in_mags)
        np.testing.assert_array_equal(out_sigmas.shape, [15, 3])

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    test_get_bands_in_x()
    unittest.main()
