"""Module containing catalog-agnostic raytracing utility functions

"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def approx_mean_kappa(fit_model, weighted_mass_sum, seed):
    """Approximate the mean kappa given the log weighted sum of halo masses

    Parameters
    ----------
    fit_model : object
        any model that has a method `predict` returning the mean and standard
        deviation of predictions as a tuple, given the input X (first argument)
        and a boolean `return_std` option set to True
    weighted_mass_sum : np.array
        log10(sum over halo masses weighted by the lensing efficiency Dds/Ds),
        where halo masses are in solar masses

    Returns
    -------
    np.array
        the simulated mean kappa, including spread, of the halos

    """
    rng = np.random.default_rng(seed)
    pred, sigma = fit_model.predict(weighted_mass_sum, return_std=True)
    mean_kappa = pred.reshape(-1) + rng.standard_normal(sigma.shape)*sigma
    return mean_kappa


def fit_gp(train_X, train_Y):
    """Fit a Gaussian process regressor

    """
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp_model = GaussianProcessRegressor(kernel=None, n_restarts_optimizer=10)
    gp_model.fit(train_X, train_Y)
    return gp_model
