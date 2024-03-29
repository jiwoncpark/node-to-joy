"""Utility methods managing inference based on a trained model

"""

import os
import sys
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy import stats, special
import emcee
import matplotlib.pyplot as plt

DEBUG = False


def get_normal_logpdf(mu, log_sigma, x,
                      bounds_lower=-np.inf,
                      bounds_upper=np.inf):
    """Evaluate the log kappa likelihood of the test set,
    log p(k_j|Omega), exactly

    Note
    ----
    Only normal likelihood supported for now.

    Returns
    -------
    np.ndarray or float
        Log PDF, of shape broadcasted across mu, log_sigma, and x

    """
    # logpdf = 0.5*(np.log(ivar + 1.e-7) - ivar*(x - mu)**2.0)
    candidate = np.array([mu, log_sigma])
    if np.any(candidate < bounds_lower):
        return -np.inf
    elif np.any(candidate > bounds_upper):
        return -np.inf
    logpdf = -log_sigma - 0.5*(x - mu)**2.0/np.exp(2.0*log_sigma)
    logpdf = logpdf - 0.5*np.log(2*np.pi)  # normalization
    assert not np.isnan(logpdf).any()
    assert not np.isinf(logpdf).any()
    return logpdf


def run_mcmc(log_prob, log_prob_kwargs, p0, n_run, n_burn, chain_path,
             run_name='mcmc',
             n_walkers=100,
             plot_chain=True,
             clear=False,
             n_cores=None):
    """Run MCMC sampling

    Parameters
    ----------
    p0 : np.array of shape `[n_walkers, n_dim]`
    n_run : int
    n_burn : int
    chain_path : os.path or str
    n_walkers : int
    plot_chain : bool

    """
    n_dim = p0.shape[1]
    n_cores = cpu_count() - 2 if n_cores is None else n_cores
    # Set up the backend
    backend = emcee.backends.HDFBackend(chain_path, name=run_name)
    if clear:
        backend.reset(n_walkers, n_dim)  # clear it in case the file already exists
    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob,
                                        kwargs=log_prob_kwargs,
                                        pool=pool, backend=backend)
        if n_burn > 0:
            print("Running MCMC...")
            state = sampler.run_mcmc(p0, n_burn)
            sampler.reset()
            sampler.run_mcmc(state, n_run, progress=True)
        else:
            sampler.run_mcmc(None, n_run, progress=True)
    if plot_chain:
        samples = sampler.get_chain(flat=True)
        get_chain_plot(samples, os.path.join(os.path.dirname(chain_path),
                                             f'mcmc_chain_{os.path.split(chain_path)[1]}.png'))


def get_chain_plot(samples, out_path='mcmc_chain.png'):
    """Plot MCMC chain

    Note
    ----
    Borrowed from https://emcee.readthedocs.io/en/stable/tutorials/line/

    """
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    n_chain, n_dim = samples.shape
    labels = ["mean", "log_sigma"]
    for i in range(2):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    if out_path is None:
        plt.show()
    else:
        fig.savefig(out_path)


def get_log_p_k_given_omega_int_kde(k_train, k_bnn, kwargs=None):
    """Evaluate the log likelihood, log p(k|Omega_int),
    using kernel density estimation (KDE) on training kappa,
    on the BNN kappa samples of test sightlines

    Parameters
    ----------
    k_train : np.array of shape `[n_train]`
        kappa in the training set
    k_bnn : np.array of shape `[n_test, n_samples]`
    kwargs : dict
        currently unused, placeholder for symmetry with analytic version

    Returns
    -------
    np.array of shape `[n_test, n_samples]`
        log p(k|Omega_int)

    """
    kde = stats.gaussian_kde(k_train, bw_method='scott')
    log_p_k_given_omega_int = kde.pdf(k_bnn.reshape(-1)).reshape(k_bnn.shape)
    log_p_k_given_omega_int = np.log(log_p_k_given_omega_int)
    assert not np.isnan(log_p_k_given_omega_int).any()
    assert not np.isinf(log_p_k_given_omega_int).any()
    return log_p_k_given_omega_int


def get_log_p_k_given_omega_int_analytic(k_train, k_bnn, interim_pdf_func):
    """Evaluate the log likelihood, log p(k|Omega_int),
    using kernel density estimation (KDE) on training kappa,
    on the BNN kappa samples of test sightlines

    Parameters
    ----------
    k_train : np.array of shape `[n_train]`
        kappa in the training set. Unused.
    k_bnn : np.array of shape `[n_test, n_samples]`
    interim_pdf_func : callable
        function that evaluates the PDF of the interim prior

    Returns
    -------
    np.array of shape `[n_test, n_samples]`
        log p(k|Omega_int)

    """
    log_p_k_given_omega_int = interim_pdf_func(k_bnn)
    log_p_k_given_omega_int = np.log(log_p_k_given_omega_int)
    assert not np.isnan(log_p_k_given_omega_int).any()
    assert not np.isinf(log_p_k_given_omega_int).any()
    return log_p_k_given_omega_int


def get_omega_post(k_bnn, log_p_k_given_omega_int, mcmc_kwargs,
                   bounds_lower, bounds_upper):
    """Sample from p(Omega|{d}) using MCMC

    Parameters
    ----------
    k_bnn : np.array of shape `[n_test, n_samples]`
        BNN samples for `n_test` sightlines
    log_p_k_given_omega_int : np.array of shape `[n_test, n_samples]`
        log p(k_bnn|Omega_int)

    """
    np.random.seed(42)
    k_bnn = k_bnn.squeeze(1)  # pop the lone Y_dim dimension
    log_p_k_given_omega_func = partial(get_normal_logpdf, x=k_bnn,
                                       bounds_lower=bounds_lower,
                                       bounds_upper=bounds_upper)
    mcmc_kwargs['log_prob_kwargs'] = dict(
                                          log_p_k_given_omega_func=log_p_k_given_omega_func,
                                          log_p_k_given_omega_int=log_p_k_given_omega_int
                                          )
    if DEBUG:
        print("int", log_p_k_given_omega_int.shape)
        print("kbnn", k_bnn.shape)
        np.save('num.npy', log_p_k_given_omega_func(0.01, np.log(0.04)))
        np.save('denom.npy', log_p_k_given_omega_int)
        p = np.zeros([200, 100])
        xx, yy = np.meshgrid(np.linspace(0, 0.05, 200),
                             np.linspace(-7, -3, 100), indexing='ij')
        for i in range(200):
            for j in range(100):
                p[i, j] = log_prob_mcmc([xx[i, j], yy[i, j]],
                                        **mcmc_kwargs['log_prob_kwargs'])
        np.save('p_look.npy', p)

    run_mcmc(log_prob_mcmc, **mcmc_kwargs)


def get_omega_post_loop(k_samples_list, log_p_k_given_omega_int_list,
                        mcmc_kwargs,
                        bounds_lower, bounds_upper):
    """Sample from p(Omega|{d}) using MCMC

    Parameters
    ----------
    k_samples_list : list
        Each element is the array of samples for a sightline, so the
        list has length `n_test`
    log_p_k_given_omega_int_list : list
        Each element is the array of log p(k_samples|Omega_int) for a
        sightline, so the list has length `n_test`

    """
    np.random.seed(42)

    n_test = len(k_samples_list)
    log_p_k_given_omega_func_list = []
    for los_i in range(n_test):
        log_p_k_given_omega_func_list.append(partial(get_normal_logpdf,
                                                     x=k_samples_list[los_i],
                                                     bounds_lower=bounds_lower,
                                                     bounds_upper=bounds_upper))
    kwargs = dict(
                  log_p_k_given_omega_func_list=log_p_k_given_omega_func_list,
                  log_p_k_given_omega_int_list=log_p_k_given_omega_int_list
                  )
    mcmc_kwargs['log_prob_kwargs'] = kwargs

    if DEBUG:
        print("n_test")
        print(len(k_samples_list), len(log_p_k_given_omega_int_list))
        print(len(log_p_k_given_omega_func_list))
        print("n_samples of first sightline")
        print(len(k_samples_list[0]), len(log_p_k_given_omega_int_list[0]))
    if DEBUG:
        np.save('func_list_num', log_p_k_given_omega_func_list)
        np.save('arr_list_denom', log_p_k_given_omega_int_list)
        
    run_mcmc(log_prob_mcmc_loop, **mcmc_kwargs)


def log_prob_mcmc(omega, log_p_k_given_omega_func, log_p_k_given_omega_int):
    """Evaluate the MCMC objective

    Parameters
    ----------
    omega : list
        Current MCMC sample of [mu, log_sigma] = Omega
    log_p_k_given_omega_func : callable
        function that returns p(k|Omega) of shape [n_test, n_samples]
        for given omega and k fixed to be the BNN samples
    log_p_k_given_omega_int : np.ndarray
        Values of p(k|Omega_int) of shape [n_test, n_samples] for k
        fixed to be the BNN samples

    Returns
    -------
    float
        Description

    """
    log_p_k_given_omega = log_p_k_given_omega_func(omega[0], omega[1])
    log_ratio = log_p_k_given_omega - log_p_k_given_omega_int  # [n_test, n_samples]
    mean_over_samples = special.logsumexp(log_ratio,
                                          # normalization
                                          b=1.0/log_ratio.shape[-1],
                                          axis=1)  # [n_test,]
    assert not np.isnan(log_p_k_given_omega_int).any()
    assert not np.isinf(log_p_k_given_omega_int).any()
    log_prob = mean_over_samples.sum()  # summed over sightlines
    # log_prob = log_prob - n_test*np.log(n_samples)  # normalization
    return log_prob


def log_prob_mcmc_loop(omega, log_p_k_given_omega_func_list,
                       log_p_k_given_omega_int_list):
    """Evaluate the MCMC objective

    Parameters
    ----------
    omega : list
        Current MCMC sample of [mu, log_sigma] = Omega
    log_p_k_given_omega_func_list : list of callable
        List of functions that returns p(k|Omega) of shape [n_samples]
        for given omega and k fixed to be the posterior samples
    log_p_k_given_omega_int_list : np.ndarray
        List of values of p(k|Omega_int) of shape [n_samples] for k
        fixed to be the posterior samples

    Returns
    -------
    float
        Description

    """
    n_test = len(log_p_k_given_omega_int_list)
    mean_over_samples_dataset = np.empty(n_test)
    for los_i in range(n_test):
        # log_p_k_given_omega ~ [n_samples]
        log_p_k_given_omega = log_p_k_given_omega_func_list[los_i](omega[0], omega[1])
        log_ratio = log_p_k_given_omega - log_p_k_given_omega_int_list[los_i]  # [n_samples]
        # print(len(log_ratio), len(log_p_k_given_omega), len(log_p_k_given_omega_int_list[los_i]))
        
        mean_over_samples = special.logsumexp(log_ratio,
                                              # normalization
                                              b=1.0/len(log_ratio))  # scalar
        if DEBUG:
            if np.sum(~np.isfinite(log_ratio)) > 0:
                print("has nan in log_ratio")
                print("number of nans: ", np.sum(~np.isfinite(log_ratio)))
                print("los:", los_i)
                print("omega:", omega)
                print("num:", log_p_k_given_omega)
                print("denom:", log_p_k_given_omega_int_list[los_i])
            if np.isnan(mean_over_samples):
                print("NaN mean over samples")
                print("los:", los_i)
                print("omega:", omega)
                print("num:", log_p_k_given_omega)
                print("denom:", log_p_k_given_omega_int_list[los_i])
        mean_over_samples_dataset[los_i] = mean_over_samples
    is_finite = np.isfinite(mean_over_samples_dataset)
    n_eff_samples = np.sum(is_finite)
    good = mean_over_samples_dataset[is_finite]
    if n_eff_samples < n_test:
        return -np.inf
        #if n_eff_samples == 0:
        #    # Happens when proposed omega is out of bounds, 
        #    # i.e. numerator `log_p_k_given_omega` is -np.inf, a scalar
        #    return -np.inf
        #else:
        #    if DEBUG:
        #        warnings.warn(f"Effective samples {n_eff_samples} less than total")
    # assert not np.isnan(mean_over_samples_dataset).any()
    # assert not np.isinf(mean_over_samples_dataset).any()
    log_prob = good.sum()  # summed over sightlines
    # log_prob = log_prob - n_test*np.log(n_samples)  # normalization
    return log_prob


def get_mcmc_samples(chain_path, chain_kwargs):
    """Load the samples from saved MCMC run

    Parameters
    ----------
    chain_path : str
        Path to the stored chain
    chain_kwargs : dict
        Options for chain postprocessing, including flat, thin, discard

    Returns
    -------
    np.array of shape `[n_omega, 2]`
        omega samples from MCMC chain

    """
    backend = emcee.backends.HDFBackend(chain_path)
    chain = backend.get_chain(**chain_kwargs)
    return chain


def get_kappa_log_weights(k_bnn, log_p_k_given_omega_int,
                          omega_post_samples=None):
    """Evaluate the log weights used to reweight individual kappa posteriors

    Parameters
    ----------
    k_bnn : np.ndarray of shape `[n_samples]`
        BNN posterior samples
    log_p_k_given_omega_int : np.ndarray of shape `[n_samples]`
        Likelihood of BNN kappa given the interim prior.
    omega_post_samples : np.ndarray, optional
        Omega posterior samples used as the prior to apply.
        Should be np.array of shape `[n_omega, 2]`.
        If None, only division by the interim prior will be done.

    Returns
    -------
    np.ndarray
        log weights evaluated at k_bnn

    """
    k_bnn = k_bnn.reshape(-1, 1)  # [n_samplses, 1]
    if omega_post_samples is not None:
        mu = omega_post_samples[:, 0].reshape([1, -1])  # [1, n_omega]
        log_sigma = omega_post_samples[:, 1].reshape([1, -1])  # [1, n_omega]
        num = get_normal_logpdf(x=k_bnn,
                                mu=mu,
                                log_sigma=log_sigma)  # [n_samples, n_omega]
    else:
        num = 0.0
    denom = log_p_k_given_omega_int[:, np.newaxis]  # [n_samples, 1]
    log_weights = special.logsumexp(num - denom, axis=-1)  # [n_samples]
    return log_weights


def get_kappa_log_weights_vectorized(k_bnn, omega_post_samples, log_p_k_given_omega_int):
    """Evaluate the log weights used to reweight individual kappa posteriors

    Parameters
    ----------
    k_bnn : np.array of shape `[n_test, n_samples]`
    omega_post_samples : np.array of shape `[n_omega, 2]`
    log_p_k_given_omega_int : np.array of shape `[n_test, n_samples]`

    """
    k_bnn = k_bnn[:, :, np.newaxis]  # [n_test, n_samples, 1]
    mu = omega_post_samples[:, 0].reshape([1, 1, -1])  # [1, 1, n_omega]
    log_sigma = omega_post_samples[:, 0].reshape([1, 1, -1])  # [1, 1, n_omega]
    num = get_normal_logpdf(x=k_bnn,
                            mu=mu,
                            log_sigma=log_sigma)  # [n_test, n_samples, n_omega]
    denom = log_p_k_given_omega_int[:, :, np.newaxis]
    weights = special.logsumexp(num - denom, axis=-1)  # [n_test, n_samples]
    return weights


def resample_from_pdf(grid, log_pdf, n_samples):
    if n_samples > len(grid):
        fine_grid = np.linspace(grid.min(), grid.max(), n_samples*5)
        pdf = np.interp(fine_grid, grid, np.exp(log_pdf))
    else:
        fine_grid = grid
        pdf = np.exp(log_pdf)
    # Normalize to unity
    pdf = pdf/np.sum(pdf)
    resampled = np.random.choice(fine_grid,
                                 size=n_samples,
                                 replace=True,
                                 p=pdf)
    return resampled


def fit_kde_on_weighted_samples(samples, weights=None):
    """Fit a KDE on weighted samples

    """
    samples = samples.squeeze()
    if weights is not None:
        weights = weights.squeeze()
    kde = stats.gaussian_kde(samples, bw_method='scott', weights=weights)
    return kde


def resample_from_samples(samples, weights, n_resamples, plot_path=None):
    """Resample from a distribution defined by weighted samples

    Parameters
    ----------
    samples : np.ndarray
    weights : np.ndarray
    n_resamples : int
    plot_path : str
        Path for the plot illustrating the KDE fit
    """
    kde = fit_kde_on_weighted_samples(samples, weights)
    samples = samples.squeeze()
    weights = weights.squeeze()
    resamples = kde.resample(n_resamples).squeeze()
    if plot_path is not None:
        grid = np.linspace(-0.2, 0.2, 50)
        plt.hist(samples, density=True,
                 histtype='step', color='tab:gray', label='orig samples')
        plt.hist(samples, weights=weights, density=True,
                 alpha=0.5, color='tab:red', label='weighted samples')
        plt.plot(grid, kde.pdf(grid), color='k', label='KDE fit')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
    return resamples

