"""Catalog-agnostic raytracing module

"""
import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import n2j.trainval_data.utils.raytracing_utils as ru

__all__ = ['BaseRaytracer']


class BaseRaytracer(ABC):
    """Base class for raytracer tools. Not to be instantiated on its own.
    Child classes inherit catalog-agnostic methods from this class.

    """

    def __init__(self,
                 in_dir: str = '.',
                 out_dir: str = '.',
                 debug: bool = False,):
        """
        Parameters
        ----------
        out_dir : str
            destination folder
        test : bool
            test mode

        """
        self.in_dir = in_dir
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        self.debug = debug
        # Paths for output data files
        self._define_paths()
        # Optionally overridden
        self.halo_satisfies = []

    def _define_paths(self):
        """Define output file paths and path formats

        """
        self.sightlines_path = os.path.join(self.out_dir, 'sightlines.npy')
        self.Y_path = os.path.join(self.out_dir, 'Y.csv')
        self.halo_path_fmt = os.path.join(self.out_dir,
                                          'halos_los={0:07d}_id={1:012d}.npy')
        self.k_map_fmt = os.path.join(self.out_dir,
                                      'k_map_los={0:07d}.npy')
        self.g1_map_fmt = os.path.join(self.out_dir,
                                       'g1_map_los={0:07d}.npy')
        self.g2_map_fmt = os.path.join(self.out_dir,
                                       'g2_map_los={0:07d}.npy')
        self.k_samples_fmt = os.path.join(self.out_dir,
                                          'k_samples_los={0:07d}.npy')
        self.k_samples_map_fmt = os.path.join(self.out_dir,
                                              'k_map_los={0:07d}_sample={1:04d}.npy')
        self.uncalib_path = os.path.join(self.out_dir, 'uncalib.csv')
        # out_dir of a separate run that had n_kappa_samples > 0
        # that can be used to estimate the mapping between weighted sum of halo
        # masses and mean of kappa samples.

    @abstractmethod
    def get_pointings_iterator(self):
        return NotImplementedError

    @abstractmethod
    def get_halos_iterator(self):
        return NotImplementedError

    def apply_calibration(self):
        """Subtract off the extra mass added when we raytraced through
        parameterized halos

        """
        Y = pd.DataFrame(np.load(self.sightlines_path),
                         columns=self.pointings_cols)  # init with pointings df
        uncalib = pd.read_csv(self.uncalib_path,
                              index_col=None,
                              ).drop_duplicates('idx').sort_values(by=['idx'])
        mean_kappas = self.get_mean_kappas()
        # To the WL quantities, add our raytracing and subtract mean mass
        final_kappa = uncalib['kappa'].values + Y['kappa'].values - mean_kappas
        final_gamma1 = uncalib['gamma1'].values + Y['gamma1'].values
        final_gamma2 = uncalib['gamma2'].values + Y['gamma2'].values
        # Also log the mean kappa contribution of halos, for fitting later
        Y['mean_kappa'] = mean_kappas
        Y['final_kappa'] = final_kappa
        Y['final_gamma1'] = final_gamma1
        Y['final_gamma2'] = final_gamma2
        # Update the Y df
        Y[self.Y_cols].to_csv(self.Y_path, index=None)

    def get_mean_kappas(self):
        """Get the mean kappa contribution of LOS halos

        """
        if self.n_kappa_samples > 0:
            return self.load_mean_kappas_from_file()
        else:
            return self.estimate_mean_kappas()

    def load_mean_kappas_from_file(self):
        """Load the mean kappa contribution of LOS halos from the saved files
        from explicit computation

        """
        pointings = pd.DataFrame(np.load(self.sightlines_path),
                                 columns=self.pointings_cols)
        n_sightlines = pointings.shape[0]
        mean_kappas = np.empty(n_sightlines)
        # Compute mean kappa of halos in each sightline
        for los_i in range(n_sightlines):
            samples = np.load(self.k_samples_fmt.format(los_i))
            samples = samples[samples < 0.5]  # remove overdense outliers
            mean_kappas[los_i] = np.mean(samples)
        return mean_kappas

    def estimate_mean_kappas(self):
        """Estimate the mean kappas from a set of "training" pairs generated in
        another run

        """
        # Fit a model using the kappa_sampling run as the training data
        uncalib_train_path = os.path.join(self.kappa_sampling_dir, 'uncalib.csv')
        X_train = pd.read_csv(uncalib_train_path,
                              usecols=['idx', 'weighted_mass_sum'],
                              index_col=None,
                              ).drop_duplicates('idx').sort_values(by=['idx'])
        X_train = X_train['weighted_mass_sum'].values.reshape(-1, 1)
        Y_train_path = os.path.join(self.kappa_sampling_dir, 'Y.csv')
        Y_train = pd.read_csv(Y_train_path,
                              usecols=['mean_kappa'],
                              index_col=None,
                              ).values
        fit_model = ru.fit_gp(X_train, Y_train)
        # Predict on this run
        X_test = pd.read_csv(self.uncalib_path,
                             usecols=['idx', 'weighted_mass_sum'],
                             index_col=None,
                             ).drop_duplicates('idx').sort_values(by=['idx'])
        X_test = X_test['weighted_mass_sum'].values.reshape(-1, 1)
        mean_kappas = ru.approx_mean_kappa(fit_model, X_test, seed=self.seed)
        if self.debug:
            import matplotlib.pyplot as plt
            plt.scatter(X_train, Y_train, color='b', label='train')
            plt.scatter(X_test, mean_kappas, color='r', label='test')
            plt.legend()
            plt.show()
            plt.savefig('mean_kappa_interp_hp={:d}.png'.format(self.healpix))
        return mean_kappas
