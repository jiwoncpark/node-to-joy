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


class ErrorSimulator:
    bands = np.array([['u', 'g', 'r', 'i', 'z', 'y']])

    # expected median zenith sky brightness
    m_skys = [22.99, 22.26, 21.20, 20.48, 19.60, 18.61]  # row 1, Table 2
    # expected median zenith seeing (FWHM, arcsec) - FIXME - should we add scaling with airmass, X (~X^0.6)?
    seeings = [0.92, 0.87, 0.83, 0.80, 0.78, 0.76]  # row 3, Table 2
    # band-dependent param, depends on sky brightness, readout noise, etc.,
    gammas = [0.038, 0.039, 0.039, 0.039, 0.039, 0.039]  # row 4, Table 2
    # adopted atmospheric extinction
    k_ms = [0.491, 0.213, 0.126, 0.096, 0.069, 0.170]  # row 5, Table 2
    # band-dependent param for calculation of 5 sigma depth, depends on throughput of optical elements and sensors
    C_ms = [23.09, 24.42, 24.44, 24.32, 24.16, 23.73]  # row 6, Table 2
    # loss of depth due to instrumental noise
    delta_C_m_infs = [0.62, 0.18, 0.10, 0.07, 0.05, 0.04]  # row 8, Table 2

    num_visits_10_year = [56, 80, 184, 184, 160, 160]  # from 2019 Science Drivers table 1

    def __init__(self, mags, depth=5, airmass=1.2):
        # mags: numpy array of shape (1, 6) representing the true magnitudes where 6 is for ugrizy
        self.mags = mags
        self.depth = depth
        self.airmass = airmass

        self.num_visits = np.ones(6)
        if depth != 'single_visit':  # assuming annual obs strategy is fixed
            self.num_visits = [(x * depth) // 10 for x in self.num_visits_10_year]

    def delta_C_m(self, band_index):
        """Returns delta_C_m correction for num_visits > 1 (i.e. exposure times > 30s),
        following Eq 7 in Science Drivers.
        """
        delta_C_m_inf = self.delta_C_m_infs[band_index]
        delta_C_m = delta_C_m_inf - 1.25 * np.log10(1 + (10 ** (0.8 * delta_C_m_inf) - 1) / self.num_visits[band_index])
        return delta_C_m

    def calculate_5sigma_depth_from_scratch(self, band_index):
        """Returns m_5 found using Eq 6 in Science Drivers,
        using eff seeing, sky brightness, exposure time, extinction coeff, airmass.
        Includes dependence on number of visits.
        """
        i = band_index

        C_m = self.C_ms[i]
        m_sky = self.m_skys[i]
        seeing = self.seeings[i]
        k_m = self.k_ms[i]
        num_visit = self.num_visits[i]

        m_5 = C_m + 0.50 * (m_sky - 21) + 2.5 * np.log10(0.7 / seeing) \
            + 1.25 * np.log10(num_visit) - k_m * (self.airmass - 1)
        m_5 += self.delta_C_m(i)

        return m_5

    def calculate_rand_err(self, band, mag):
        """"Returns sigma_rand_squared"""
        all_bands = 'ugrizy'
        band_index = all_bands.find(band)

        m_5 = self.calculate_5sigma_depth_from_scratch(band_index)
        # print("m5", m_5, "band_index", band_index)
        x = 10.0 ** (0.4 * (mag - m_5))
        # print("x", x)

        gamma = self.gammas[band_index]
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
        return np.sqrt(sigma_sys ** 2 + sigma_rand_squared)  # adding errors in quadrature

    def get_sigmas(self):
        calculate_photo_err_v = np.vectorize(self.calculate_photo_err)

        sigmas = calculate_photo_err_v(self.bands, self.mags)
        return sigmas
        # FIXME: take out vectorization

    def __call__(self):
        sigmas = self.get_sigmas()
        return self.mags + np.random.normal(loc=np.zeros((1, 6)), scale=sigmas)
