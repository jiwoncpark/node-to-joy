"""Utility methods managing inference based on a trained model

"""

import os
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy import stats, special
import emcee
import matplotlib.pyplot as plt


def get_normal_logpdf(mu, log_sigma, x,
                      bounds_lower=-np.inf,
                      bounds_upper=np.inf):
    """Evaluate the log kappa likelihood of the test set,
    log p(k_j|Omega), exactly

    Note
    ----
    Only normal likelihood supported for now.

    """
    # logpdf = 0.5*(np.log(ivar + 1.e-7) - ivar*(x - mu)**2.0)
    candidate = np.array([mu, log_sigma])
    if np.any(candidate < bounds_lower):
        return -np.inf
    elif np.any(candidate > bounds_upper):
        return -np.inf
    logpdf = -log_sigma - 0.5*(x - mu)**2.0/np.exp(2.0*log_sigma)
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
            state = sampler.run_mcmc(p0, n_burn)
            sampler.reset()
            sampler.run_mcmc(state, n_run, progress=True)
        else:
            sampler.run_mcmc(None, n_run, progress=True)
    if plot_chain:
        samples = sampler.get_chain(flat=True)
        get_chain_plot(samples, os.path.join(os.path.dirname(chain_path),
                                             'mcmc_chain.png'))


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
    fig.savefig(out_path)


def get_log_p_k_given_omega_int(k_train, k_bnn):
    """Evaluate the log likelihood, log p(k|Omega_int),
    using kernel density estimation (KDE) on training kappa,
    on the BNN kappa samples of test sightlines

    Parameters
    ----------
    k_train : np.array of shape `[n_train]`
        kappa in the training set
    k_bnn : np.array of shape `[n_test, n_samples]`

    Returns
    -------
    np.array of shape `[n_test, n_samples]`
        log p(k|Omega_int)

    """
    kde = stats.gaussian_kde(k_train, bw_method='scott')
    log_p_k_given_omega_int = kde.pdf(k_bnn.reshape(-1)).reshape(k_bnn.shape)
    log_p_k_given_omega_int = np.log(log_p_k_given_omega_int + 1.e-7)
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
    log_p_k_given_omega_func = partial(get_normal_logpdf, x=k_bnn,
                                       bounds_lower=bounds_lower,
                                       bounds_upper=bounds_upper)
    mcmc_kwargs['log_prob_kwargs'] = dict(
                                          log_p_k_given_omega_func=log_p_k_given_omega_func,
                                          log_p_k_given_omega_int=log_p_k_given_omega_int
                                          )
    run_mcmc(log_prob_mcmc, **mcmc_kwargs)


def log_prob_mcmc(omega, log_p_k_given_omega_func, log_p_k_given_omega_int):
    """Evaluate the MCMC objective

    """
    log_p_k_given_omega = log_p_k_given_omega_func(omega[0], omega[1])
    log_ratio = log_p_k_given_omega - log_p_k_given_omega_int  # [n_test, n_samples]
    summed_over_samples = special.logsumexp(log_ratio, axis=1)  # [n_test,]
    assert not np.isnan(log_p_k_given_omega_int).any()
    assert not np.isinf(log_p_k_given_omega_int).any()
    return summed_over_samples.mean()  # averaged over sightlines


def get_mcmc_samples(chain_path, chain_kwargs):
    """Load the samples from saved MCMC run

    Returns
    -------
    np.array of shape `[n_omega, 2]`
        omega samples from MCMC chain

    """
    backend = emcee.backends.HDFBackend(chain_path)
    chain = backend.get_chain(**chain_kwargs)
    return chain


def get_kappa_log_weights(k_bnn, omega_post_samples, log_p_k_given_omega_int):
    """Evaluate the log weights used to reweight individual kappa posteriors

    Parameters
    ----------
    k_bnn : np.array of shape `[n_samples]`
    omega_post_samples : np.array of shape `[n_omega, 2]`
    log_p_k_given_omega_int : np.array of shape `[n_samples]`

    """
    k_bnn = k_bnn[:, np.newaxis]  # [n_samples, 1]
    mu = omega_post_samples[:, 0].reshape([1, -1])  # [1, n_omega]
    log_sigma = omega_post_samples[:, 1].reshape([1, -1])  # [1, n_omega]
    num = get_normal_logpdf(x=k_bnn,
                            mu=mu,
                            log_sigma=log_sigma)  # [n_samples, n_omega]
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
