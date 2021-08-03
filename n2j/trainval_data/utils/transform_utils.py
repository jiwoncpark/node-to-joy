"""Utility functions for processing data used for training and validation

"""
import numpy as np


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class Slicer:
    def __init__(self, feature_idx):
        self.feature_idx = feature_idx

    def __call__(self, x):
        return x[:, self.feature_idx]


class MagErrorSimulator:
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    # expected median zenith sky brightness
    # tuned to DC2 (u, z, y bands changed)
    m_skys = [22.5, 22.19191, 21.10172, 19.93964, 18.3, 17.7]
    # OpSim filtSkyBrightness medians
    # m_skys = [22.71329, 22.19191, 21.10172, 19.93964, 19.00231, 18.24634]
    # row 1, Table 2
    # m_skys = [22.99, 22.26, 21.20, 20.48, 19.60, 18.61]

    # expected median zenith seeing (FWHM, arcsec)
    # tuned to DC2 (z, y bands changed)
    seeings = [1.029668, 0.951018, 0.8996875, 0.868422, 1, 1]
    # OpSim FWHMeff medians
    # seeings = [1.029668, 0.951018, 0.8996875, 0.868422, 0.840263, 0.8486855]
    # row 3, Table 2
    # seeings = [0.92, 0.87, 0.83, 0.80, 0.78, 0.76]

    # band-dependent param, depends on sky brightness, readout noise, etc.
    # row 4, Table 2
    gammas = [0.038, 0.039, 0.039, 0.039, 0.039, 0.039]

    # adopted atmospheric extinction
    # row 5, Table 2
    k_ms = [0.491, 0.213, 0.126, 0.096, 0.069, 0.170]

    # band-dependent param for calculation of 5 sigma depth,
    # depends on throughput of optical elements and sensors
    # row 6, Table 2
    C_ms = [23.09, 24.42, 24.44, 24.32, 24.16, 23.73]

    # loss of depth due to instrumental noise
    # row 8, Table 2
    delta_C_m_infs = [0.62, 0.18, 0.10, 0.07, 0.05, 0.04]

    # from 2019 Science Drivers table 1
    num_visits_10_year = [56, 80, 184, 184, 160, 160]

    def __init__(self,
                 mag_idx=[2, 3, 4, 5, 6, 7],
                 which_bands=list('ugrizy'),
                 depth=5,
                 airmass=1.15304):  # median OpSimairmass
        """
        Parameters
        ----------
        mag_idx: list
            indices of ugrizy mags in x
        which_bands: list
            bands corresponding to mag_idx.
            Must be same length as `mag_idx`
        depth: float
            LSST survey depth in years or "single_visit"

        """
        self.mag_idx = mag_idx
        self.which_bands = which_bands
        self.depth = depth
        self.airmass = airmass
        self._format_input_params()

        if depth == 'single_visit':
            self.num_visits = np.ones((1, 6))
        else:
            # Downscale number of visits based on 10-year depth
            # assuming annual obs strategy is fixed
            self.num_visits = self.num_visits_10_year*depth//10

    def _format_input_params(self):
        """Convert param lists into arrays for vectorized computation

        """
        params_list = ['bands', 'm_skys', 'seeings', 'gammas']
        params_list += ['k_ms', 'C_ms', 'delta_C_m_infs']
        params_list += ['num_visits_10_year']
        for key in params_list:
            val = np.array(getattr(self, key)).reshape([1, -1])
            setattr(self, key, val)

    def calculate_delta_C_m(self, band_index):
        """Returns delta_C_m correction for num_visits > 1 (i.e. exposure times > 30s),
        following Eq 7 in Science Drivers.
        """
        delta_C_m_inf = self.delta_C_m_infs[0, band_index]
        delta_C_m = delta_C_m_inf - 1.25 * np.log10(1 + (10 ** (0.8 * delta_C_m_inf) - 1) / self.num_visits[0, band_index])
        return delta_C_m

    def calculate_5sigma_depth_from_scratch(self, band_index):
        """Returns m_5 found using Eq 6 in Science Drivers,
        using eff seeing, sky brightness, exposure time, extinction coeff, airmass.
        Includes dependence on number of visits.
        """
        i = band_index

        C_m = self.C_ms[0, i]
        m_sky = self.m_skys[0, i]
        seeing = self.seeings[0, i]
        k_m = self.k_ms[0, i]
        num_visit = self.num_visits[0, i]

        m_5 = C_m + 0.50 * (m_sky - 21) + 2.5 * np.log10(0.7 / seeing) \
            + 1.25 * np.log10(num_visit) - k_m * (self.airmass - 1)
        # print("m_5 pre delta_C_m", m_5)
        # print("delta_C_m(i)", self.delta_C_m(i))
        m_5 += self.calculate_delta_C_m(i)

        return m_5

    def calculate_rand_err(self, band, mag):
        """"Returns sigma_rand_squared"""
        all_bands = 'ugrizy'
        band_index = all_bands.find(band)

        m_5 = self.calculate_5sigma_depth_from_scratch(band_index)
        # print("m5", m_5, "band_index", band_index)
        x = 10.0 ** (0.4 * (mag - m_5))
        # print("x", x)

        gamma = self.gammas[0, band_index]
        # print("gamma", gamma)
        sigma_rand_squared = (0.04 - gamma) * x + gamma * (x ** 2)

        return sigma_rand_squared

    def calculate_photo_err(self, band, mag):
        """"Returns sigma (photometric error in mag).
        Calculated using figures and formulae from Science Drivers

        Params:
        - band (str) ['u', 'g', 'r', 'i', 'z', 'y']
        - AB mag (float)
        - depth int in range(1, 11) or 'single_visit'
        """
        # random photometric error
        sigma_rand_squared = self.calculate_rand_err(band, mag)
        # print("rand portion", sigma_rand_squared**0.5)

        # systematic photometric error
        sigma_sys = 0.005
        return np.sqrt(sigma_sys**2 + sigma_rand_squared)  # adding errors in quadrature

    def get_sigmas(self, mags):
        calculate_photo_err_v = np.vectorize(self.calculate_photo_err)

        sigmas = calculate_photo_err_v(self.bands, mags)
        return sigmas
        # FIXME: take out np.vectorize()

    def __call__(self, x):
        mags = x[:, self.mag_idx]
        # mags: numpy array of shape (batch_size, 6)
        # representing the true magnitudes where 6 is for ugrizy

        sigmas = self.get_sigmas(mags)
        mags += np.random.normal(loc=np.zeros((1, 6)), scale=sigmas)  # FIXME: loc shape
        x[:, self.mag_idx] = mags
        return x

        # FIXME: don't assume we have all ugrizy mags - add support for Nones
