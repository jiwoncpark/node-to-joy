"""Utility functions for processing data used for training and validation

"""
import numpy as np
import torch


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


def get_bands_in_x(x_cols):
    mag_idx = []
    which_bands = []
    for bp in ['u', 'g', 'r', 'i', 'z', 'Y']:
        mag_col = f'mag_{bp}_lsst'  # column name for mag in bp
        if mag_col in x_cols:
            which_bands.append(bp.lower())  # this bp was observed
            idx_in_x_cols = x_cols.index(mag_col)  # index in x's dim 1
            mag_idx.append(idx_in_x_cols)
    return mag_idx, which_bands


class MagErrorSimulator:
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    # Expected median zenith sky brightness
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

    # Madi TODO: paste what this means from the paper
    # band-dependent param, depends on sky brightness, readout noise, etc.
    # row 4, Table 2
    gammas = [0.038, 0.039, 0.039, 0.039, 0.039, 0.039]

    # Adopted atmospheric extinction
    # row 5, Table 2
    k_ms = [0.491, 0.213, 0.126, 0.096, 0.069, 0.170]

    # Band-dependent param for calculation of 5 sigma depth,
    # depends on throughput of optical elements and sensors
    # row 6, Table 2
    C_ms = [23.09, 24.42, 24.44, 24.32, 24.16, 23.73]

    # Madi TODO: where "inf" comes from
    # Loss of depth due to instrumental noise
    # row 8, Table 2
    delta_C_m_infs = [0.62, 0.18, 0.10, 0.07, 0.05, 0.04]

    # from 2019 Science Drivers table 1
    num_visits_10_year = [56, 80, 184, 184, 160, 160]

    def __init__(self,
                 mag_idx=[2, 3, 4, 5, 6, 7],
                 which_bands=list('ugrizy'),
                 override_kwargs=None,
                 depth=5,
                 airmass=1.15304):  # median OpSim airmass
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
        override_kwargs: dict
            band-dependent parameters to overwrite

        """
        self.mag_idx = mag_idx
        self.which_bands = which_bands
        self.idx_in_ugrizy = [self.bands.index(b) for b in self.which_bands]
        self.depth = depth
        self.airmass = airmass
        # Overwrite band-dependent params
        if override_kwargs is not None:
            for k, v in override_kwargs.items():
                setattr(self, k, v)
        # Format them for vectorization
        self._format_input_params()

        if depth == 'single_visit':
            self.num_visits = np.ones((1, 6))
        else:
            # Downscale number of visits based on 10-year depth
            # assuming annual obs strategy is fixed
            self.num_visits = self.num_visits_10_year*depth//10
        # Precompute derivative params
        self.delta_C_ms = self.calculate_delta_C_ms()
        self.m_5s = self.calculate_5sigma_depths()
        self._slice_input_params()

    def _format_input_params(self):
        """Convert param lists into arrays for vectorized computation

        """
        params_list = ['m_skys', 'seeings', 'gammas']
        params_list += ['k_ms', 'C_ms', 'delta_C_m_infs']
        params_list += ['num_visits_10_year']
        for key in params_list:
            val = np.array(getattr(self, key)).reshape([1, -1])
            setattr(self, key, val)

    def _slice_input_params(self):
        """Slice and reorder input params so only the relevant bands
        in `self.which_bands` remain, in that order

        """
        params_list = ['m_skys', 'seeings', 'gammas']
        params_list += ['k_ms', 'C_ms', 'delta_C_m_infs']
        params_list += ['num_visits_10_year', 'delta_C_ms', 'm_5s']
        for key in params_list:
            val = getattr(self, key)[:, self.idx_in_ugrizy]
            setattr(self, key, val)

    def calculate_delta_C_ms(self):
        """Returns delta_C_m correction for num_visits > 1
        (i.e. exposure times > 30s), for ugrizy
        following Eq 7 in Science Drivers.

        """
        delta_C_ms = self.delta_C_m_infs
        to_log = 1 + (10**(0.8*self.delta_C_m_infs) - 1)/self.num_visits
        delta_C_ms -= 1.25*np.log10(to_log)
        return delta_C_ms

    def calculate_5sigma_depths(self):
        """Returns m_5 found using Eq 6 in Science Drivers,
        using eff seeing, sky brightness, exposure time,
        extinction coeff, airmass, for ugrizy.
        Includes dependence on number of visits.

        """
        m_5s = self.C_ms + 0.50*(self.m_skys - 21) \
            + 2.5*np.log10(0.7/self.seeings) \
            + 1.25*np.log10(self.num_visits) \
            - self.k_ms*(self.airmass - 1) \
            + self.delta_C_ms
        return m_5s

    def calculate_rand_err(self, mags):
        """"Returns sigma_rand_squared"""
        x = 10.0 ** (0.4 * (mags - self.m_5s))
        sigma_rand_squared = (0.04 - self.gammas)*x + self.gammas*(x**2)
        return sigma_rand_squared

    def get_sigmas(self, mags):
        """"Returns sigma (photometric error in mag).
        Calculated using figures and formulae from Science Drivers

        Params:
        - AB mag (float)
        """
        # random photometric error
        sigma_rand_squared = self.calculate_rand_err(mags)
        # print("rand portion", sigma_rand_squared**0.5)

        # systematic photometric error
        sigma_sys = 0.005
        # adding errors in quadrature
        return np.sqrt(sigma_sys**2 + sigma_rand_squared)

    def __call__(self, x):
        # true magnitudes
        mags = x[:, self.mag_idx]  # shape [n_nodes, len(self.mag_idx)]
        sigmas = self.get_sigmas(mags)
        mags += np.random.normal(loc=0.0, scale=sigmas, size=mags.shape)
        x[:, self.mag_idx] = mags
        return x


class MagErrorSimulatorTorch:
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    # Expected median zenith sky brightness
    # tuned to DC2 (u, z, y bands changed)
    m_skys = [22.5, 22.19191, 21.10172, 19.93964, 18.3, 17.7]
    # OpSim filtSkyBrightness medians
    # m_skys = [22.71329, 22.19191, 21.10172, 19.93964, 19.00231, 18.24634]
    # row 1, Table 2
    # m_skys = [22.99, 22.26, 21.20, 20.48, 19.60, 18.61]

    # Expected median zenith seeing (FWHM, arcsec)
    # tuned to DC2 (z, y bands changed)
    seeings = [1.029668, 0.951018, 0.8996875, 0.868422, 1, 1]
    # OpSim FWHMeff medians
    # seeings = [1.029668, 0.951018, 0.8996875, 0.868422, 0.840263, 0.8486855]
    # row 3, Table 2
    # seeings = [0.92, 0.87, 0.83, 0.80, 0.78, 0.76]

    # Band-dependent param used in sigma_rand calculation
    # depends on sky brightness, readout noise, etc.
    # row 4, Table 2
    gammas = [0.038, 0.039, 0.039, 0.039, 0.039, 0.039]

    # Adopted atmospheric extinction
    # row 5, Table 2
    k_ms = [0.491, 0.213, 0.126, 0.096, 0.069, 0.170]

    # Band-dependent param for calculation of 5 sigma depth,
    # depends on throughput of optical elements and sensors
    # row 6, Table 2
    C_ms = [23.09, 24.42, 24.44, 24.32, 24.16, 23.73]

    # Loss of depth due to instrumental noise in a single visit
    # compared to infinite exposure time
    # row 8, Table 2
    delta_C_m_infs = [0.62, 0.18, 0.10, 0.07, 0.05, 0.04]

    # from 2019 Science Drivers table 1
    num_visits_10_year = [56, 80, 184, 184, 160, 160]

    def __init__(self,
                 mag_idx=[2, 3, 4, 5, 6, 7],
                 which_bands=list('ugrizy'),
                 override_kwargs=None,
                 depth=5,
                 airmass=1.15304):  # median OpSim airmass
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
        override_kwargs: dict
            band-dependent parameters to overwrite

        """
        self.mag_idx = mag_idx
        self.which_bands = which_bands
        self.idx_in_ugrizy = [self.bands.index(b) for b in self.which_bands]
        self.depth = depth
        self.airmass = airmass
        # Overwrite band-dependent params
        if override_kwargs is not None:
            for k, v in override_kwargs.items():
                setattr(self, k, v)
        # Format them for vectorization
        self._format_input_params()

        if depth == 'single_visit':
            self.num_visits = torch.ones((1, 6))
        else:
            # Downscale number of visits based on 10-year depth
            # assuming annual obs strategy is fixed
            self.num_visits = torch.round(self.num_visits_10_year*depth/10.0)
        # Precompute derivative params
        self.delta_C_ms = self.calculate_delta_C_ms()
        self.m_5s = self.calculate_5sigma_depths()
        self._slice_input_params()

    def _format_input_params(self):
        """Convert param lists into arrays for vectorized computation

        """
        params_list = ['m_skys', 'seeings', 'gammas']
        params_list += ['k_ms', 'C_ms', 'delta_C_m_infs']
        params_list += ['num_visits_10_year']
        for key in params_list:
            val = torch.tensor(getattr(self, key)).reshape([1, -1])
            setattr(self, key, val)

    def _slice_input_params(self):
        """Slice and reorder input params so only the relevant bands
        in `self.which_bands` remain, in that order

        """
        params_list = ['m_skys', 'seeings', 'gammas']
        params_list += ['k_ms', 'C_ms', 'delta_C_m_infs']
        params_list += ['num_visits_10_year', 'delta_C_ms', 'm_5s']
        for key in params_list:
            val = getattr(self, key)[:, self.idx_in_ugrizy]
            setattr(self, key, val)

    def calculate_delta_C_ms(self):
        """Returns delta_C_m correction for num_visits > 1
        (i.e. exposure times > 30s), for ugrizy
        following Eq 7 in Science Drivers.

        """
        delta_C_ms = self.delta_C_m_infs
        to_log = 1 + (10**(0.8*self.delta_C_m_infs) - 1)/self.num_visits
        delta_C_ms -= 1.25*torch.log10(to_log)
        return delta_C_ms

    def calculate_5sigma_depths(self):
        """Returns m_5 found using Eq 6 in Science Drivers,
        using eff seeing, sky brightness, exposure time,
        extinction coeff, airmass, for ugrizy.
        Includes dependence on number of visits.

        """
        m_5s = self.C_ms + 0.50*(self.m_skys - 21) \
            + 2.5*torch.log10(0.7/self.seeings) \
            + 1.25*torch.log10(self.num_visits) \
            - self.k_ms*(self.airmass - 1) \
            + self.delta_C_ms
        return m_5s

    def calculate_rand_err(self, mags):
        """"Returns sigma_rand_squared"""
        x = 10.0 ** (0.4 * (mags - self.m_5s))
        sigma_rand_squared = (0.04 - self.gammas)*x + self.gammas*(x**2)
        return sigma_rand_squared

    def get_sigmas(self, mags):
        """"Returns sigma (photometric error in mag).
        Calculated using figures and formulae from Science Drivers

        Params:
        - AB mag (float)
        """
        # random photometric error
        sigma_rand_squared = self.calculate_rand_err(mags)
        # print("rand portion", sigma_rand_squared**0.5)

        # systematic photometric error
        sigma_sys = 0.005
        # adding errors in quadrature
        return (sigma_sys**2 + sigma_rand_squared)**0.5

    def __call__(self, x):
        # true magnitudes
        mags = x[:, self.mag_idx]  # shape [n_nodes, len(self.mag_idx)]
        sigmas = self.get_sigmas(mags)
        mags += torch.normal(mean=torch.zeros_like(mags),
                             std=sigmas)
        x[:, self.mag_idx] = mags
        return x
