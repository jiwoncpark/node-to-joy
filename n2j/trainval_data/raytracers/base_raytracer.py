"""Catalog-agnostic raytracing module

"""
import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from n2j.trainval_data import raytracing_utils as ru

__all__ = ['BaseRaytracer']


class BaseRaytracer(ABC):
    """Base class for raytracer tools. Not to be instantiated on its own.
    Child classes inherit catalog-agnostic methods from this class.

    """

    def __init__(self,
                 out_dir: str = '.',
                 test: bool = False,):
        """
        Parameters
        ----------
        out_dir : str
            destination folder
        test : bool
            test mode

        """
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.test = test
        # Paths for output data files
        self._define_paths()
        # Optionally overridden
        self.halo_satisfies = []

    def _define_paths(self):
        """Define output file paths and path formats

        """
        self.sightlines_path = os.path.join(self.out_dir, 'sightlines.csv')
        self.halo_path_fmt = os.path.join(self.out_dir,
                                          'los_halos_los={:d}.csv')
        self.k_map_fmt = os.path.join(self.out_dir,
                                      'k_map_los={:d}.npy')
        self.g1_map_fmt = os.path.join(self.out_dir,
                                       'g1_{:d}_map_los={:d}.npy')
        self.g2_map_fmt = os.path.join(self.out_dir,
                                       'g2_{:d}_map_los={:d}.npy')
        self.k_samples_fmt = os.path.join(self.out_dir,
                                          'k_samples_los={:d}.npy')
        self.uncalib_path = os.path.join(self.out_dir, 'uncalib.csv')
        # out_dir of a separate run that had n_kappa_samples > 0
        # that can be used to estimate the mapping between weighted sum of halo
        # masses and mean of kappa samples.
        self.kappa_sampling_dir = 'kappa_sampling'

    @abstractmethod
    def get_pointings_iterator(self):
        return NotImplementedError

    @abstractmethod
    def get_gals_iterator(self):
        return NotImplementedError

    @abstractmethod
    def get_halos_iterator(self):
        return NotImplementedError

    def apply_calibration(self):
        """Subtract off the extra mass added when we raytraced through
        parameterized halos

        """
        sightlines = pd.read_csv(self.sightlines_path, index_col=None)
        uncalib = pd.read_csv(self.uncalib_path,
                              index_col=None).drop_duplicates('idx')
        mean_kappas = self.get_mean_kappas()
        # To the WL quantities, add our raytracing and subtract mean mass
        sightlines['final_kappa'] = uncalib['kappa'] + sightlines['kappa'] - mean_kappas
        sightlines['final_gamma1'] = uncalib['gamma1'] + sightlines['gamma1']
        sightlines['final_gamma2'] = uncalib['gamma2'] + sightlines['gamma2']
        # Also log the mean kappa contribution of halos, for fitting later
        sightlines['mean_kappa'] = mean_kappas
        # Update the sightlines df
        sightlines.to_csv(self.sightlines_path, index=None)

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
        sightlines = pd.read_csv(self.sightlines_path, index_col=None)
        n_sightlines = sightlines.shape[0]
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
        #import matplotlib.pyplot as plt
        if not os.path.exists(self.kappa_sampling_dir):
            raise OSError("If kappas were not sampled for each sightline,"
                          " you must generate some pairs of weighted sum of"
                          " masses and mean of kappas, in a run with out_dir"
                          " named 'kappa_sampling'.")
        # Fit a model using the kappa_sampling run as the training data
        train_uncalib_path = os.path.join(self.kappa_sampling_dir,
                                          'uncalib.csv')
        train_sightlines_path = os.path.join(self.kappa_sampling_dir,
                                             'sightlines.csv')
        train_uncalib = pd.read_csv(train_uncalib_path,
                                    index_col=None).drop_duplicates('idx')
        train_sightlines = pd.read_csv(train_sightlines_path, index_col=None)
        X_train = train_uncalib['weighted_mass_sum'].values
        Y_train = train_sightlines['mean_kappa'].values
        fit_model = ru.fit_gp(X_train.reshape(-1, 1), Y_train)
        # Predict on this run
        test_uncalib = pd.read_csv(self.uncalib_path,
                                   index_col=None).drop_duplicates('idx')
        X_test = test_uncalib['weighted_mass_sum'].values
        mean_kappas = ru.approx_mean_kappa(fit_model, X_test.reshape(-1, 1))
        #plt.scatter(X_train, Y_train, color='b', label='train')
        #plt.scatter(X_test, mean_kappas, color='r', label='test')
        #plt.legend()
        #plt.show()
        return mean_kappas
