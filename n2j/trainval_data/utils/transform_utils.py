"""Utility functions for processing data used for training and validation

"""
import numpy as np
import torch


class ComposeXYLocal:
    def __init__(self,
                 transforms_X_pre=[],
                 transforms_Y_local_pre=[],
                 transforms_X_Y_local=[],
                 transforms_X_post=[],
                 transforms_Y_local_post=[],
                 transforms_X_meta_post=[]):
        self.transforms_X_pre = transforms_X_pre
        self.transforms_Y_local_pre = transforms_Y_local_pre
        self.transforms_X_Y_local = transforms_X_Y_local
        self.transforms_X_post = transforms_X_post
        self.transforms_Y_local_post = transforms_Y_local_post
        self.transforms_X_meta_post = transforms_X_meta_post

    def __call__(self, x, y_local, x_meta):
        for t in self.transforms_X_pre:
            x = t(x)
        for t in self.transforms_Y_local_pre:
            y_local = t(y_local)
        for t_joint in self.transforms_X_Y_local:
            x, y_local = t_joint(x, y_local)
        for t in self.transforms_X_post:
            x = t(x)
        for t in self.transforms_Y_local_post:
            y_local = t(y_local)
        for t in self.transforms_X_meta_post:
            x_meta, x = t(x_meta, x)
        return x, y_local, x_meta


def get_idx(orig_list, sub_list):
    idx = []
    for item in sub_list:
        idx.append(orig_list.index(item))
    return idx


class Metadata:
    def __init__(self, 
                 all_features=['ra_true', 'dec_true', 'u', 'g', 'r', 'i', 'z', 'y'], 
                 pos_features=['ra_true', 'dec_true']):
        """Transform class for computing metadata based on transformed X.
        Metadata are unweighted and weighted number counts, where the weights are
        inverse distances.

        Parameters
        ----------
        all_features : list, optional
            All the existing features, including position features.
            Warning: must be the list after slicing! 
        pos_features : list, optional
            Euclidean-ized position features to compute metadata with.
            Warning: any set of transforms, including normalization,
            is already applied to the positions, so these aren't in physical units.
            
        """
        self.pos_idx = get_idx(all_features, pos_features)
    def __call__(self, x_meta, x):
        # Note: x_meta argument is unused.
        n_nodes = x.shape[0]
        dist = torch.sum(x[:, self.pos_idx]**2.0, axis=1)**0.5  # [n_nodes,]
        x_meta = torch.FloatTensor([[n_nodes, torch.sum(1.0/(dist + 1.e-5))]]).to(x.device)
        # Return x as second in tuple to maintain consistency
        return x_meta, x


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, y=None):
        # Second dummy argument exists for x_meta
        return (x - self.mean) / self.std


class Slicer:
    def __init__(self, feature_idx):
        self.feature_idx = feature_idx

    def __call__(self, x):
        return x[:, self.feature_idx]


class Rejector:
    def __init__(self, all_features=[],
                 ref_features=[],
                 max_vals=None,
                 min_vals=None):
        """Transform class for rejecting nodes based on feature value

        Parameters
        ----------
        all_features : list, optional
            All the existing features, including reference features to
            base rejection on
        ref_features : list, optional
            Reference features to base rejection on
        max_vals : None, optional
            Maximum allowed values for the ref_features.
            Must be same length as feature_idx
        min_vals : None, optional
            Minimum allowed values for the ref_features
            Must be same length as feature_idx
        """
        self.feature_idx = get_idx(all_features, ref_features)
        self.n_features = len(self.feature_idx)
        if self.n_features > 0:
            if max_vals is None:
                max_vals = [None]*self.n_features
            if min_vals is None:
                min_vals = [None]*self.n_features
            assert len(min_vals) == self.n_features
            assert len(max_vals) == self.n_features
            # Handle null values within provided lists
            max_vals = [float('inf') if v is None else v for v in max_vals]
            min_vals = [float('-inf') if v is None else v for v in min_vals]
            # Convert into torch tensors with broadcastable shape
            self.max_vals = torch.tensor(max_vals).reshape([1, self.n_features])
            self.min_vals = torch.tensor(min_vals).reshape([1, self.n_features])

    def __call__(self, x, y):
        if self.n_features == 0:
            return x, y  # do nothing
        ref = x[:, self.feature_idx]  # [B, n_features]
        mask = torch.logical_and(ref < self.max_vals,
                                 ref > self.min_vals)  # [B, n_features]
        mask = mask.all(dim=1)  # [B,]
        return x[mask, :], y[mask, :]


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
