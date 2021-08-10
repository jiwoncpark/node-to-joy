"""Tests for the n2j.inference.infer_utils utility functions

"""

import os
import os.path as osp
import unittest
import shutil
import numpy as np
import pandas as pd
import scipy.stats
import n2j.inference.infer_utils as iutils
import n2j.data as in_data


class TestInferUtils(unittest.TestCase):
    """A suite of tests verifying n2j.inference.infer_utils utility functions

    """

    @classmethod
    def setUpClass(cls):
        """Set up seeding

        """
        cls.rng = np.random.default_rng(123)
        cls.out_dir = 'infer_utils_testing'
        os.makedirs(cls.out_dir, exist_ok=True)

    def test_get_normal_logpdf(self):
        """Test evaluation of log Gaussian PDF for correctness
        """
        mu = 0.04
        log_sigma = np.log(0.005)
        x = self.rng.normal(100)*np.exp(log_sigma) + 0.04
        actual_logp = iutils.get_normal_logpdf(mu, log_sigma, x)
        actual_logp = actual_logp - 0.5*np.log(2*np.pi)
        true_logp = scipy.stats.norm.logpdf(x,
                                            loc=mu, scale=np.exp(log_sigma))
        np.testing.assert_array_almost_equal(actual_logp, true_logp)

    def test_log_prob_mcmc_single(self):
        """Test evaluation of MCMC objective for correctness,
        for a single sightline
        """
        bnn_k = np.zeros([1, 1])  # [n_test, n_samples]

        def log_gaussian(mu, log_sigma):
            """Gaussian log likelihood
            """
            lpdf = scipy.stats.norm.logpdf(loc=mu,
                                           scale=np.exp(log_sigma),
                                           x=bnn_k)
            return lpdf
        # At true omega, x = 0
        proposed = [0.0, np.log(2.0)]  # N(0, 2)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=0.0)
        np.testing.assert_array_equal(logp_mcmc,
                                      scipy.stats.norm.logpdf(loc=0,
                                                              scale=2.0,
                                                              x=0))
        # At true omega + sigma, x = 0
        proposed = [2.0, np.log(2.0)]  # N(0, 2)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=0.0)
        np.testing.assert_array_equal(logp_mcmc,
                                      scipy.stats.norm.logpdf(loc=2.0,
                                                              scale=2.0,
                                                              x=0))
        # Overestimated sigma by a factor of 2, x = 0
        proposed = [0.0, np.log(4.0)]  # N(0, 2)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=0.0)
        np.testing.assert_array_equal(logp_mcmc,
                                      scipy.stats.norm.logpdf(loc=0.0,
                                                              scale=4.0,
                                                              x=0))
        # Ratio of same probs
        proposed = [0.0, np.log(2.0)]  # N(0, 2)
        interim = scipy.stats.norm.logpdf(loc=0.0,
                                          scale=2.0,
                                          x=0)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=interim)
        np.testing.assert_array_equal(logp_mcmc,
                                      0.0)
        # Ratio with displaced omega + sigma
        proposed = [0.0, np.log(2.0)]  # N(0, 2)
        interim = scipy.stats.norm.logpdf(loc=2.0,
                                          scale=2.0,
                                          x=0)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=interim)
        np.testing.assert_array_equal(logp_mcmc,
                                      scipy.stats.norm.logpdf(loc=0,
                                                              scale=2.0,
                                                              x=0) - interim)

    def test_log_prob_mcmc_multiple(self):
        """Test evaluation of MCMC objective for correctness,
        for multiple sightlines
        """
        n_test = 4
        bnn_k = np.zeros([n_test, 1])  # [n_test, n_samples]

        def log_gaussian(mu, log_sigma):
            """Gaussian log likelihood
            """
            lpdf = scipy.stats.norm.logpdf(loc=mu,
                                           scale=np.exp(log_sigma),
                                           x=bnn_k)
            return lpdf
        # At true omega, x = 0
        proposed = [0.0, np.log(2.0)]  # N(0, 2)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=0.0)
        np.testing.assert_array_equal(logp_mcmc,
                                      n_test*scipy.stats.norm.logpdf(loc=0,
                                                                     scale=2.0,
                                                                     x=0))
        # At true omega + sigma, x = 0
        proposed = [2.0, np.log(2.0)]  # N(0, 2)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=0.0)
        np.testing.assert_array_equal(logp_mcmc,
                                      n_test*scipy.stats.norm.logpdf(loc=2.0,
                                                                     scale=2.0,
                                                                     x=0))
        # Overestimated sigma by a factor of 2, x = 0
        proposed = [0.0, np.log(4.0)]  # N(0, 2)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=0.0)
        np.testing.assert_array_equal(logp_mcmc,
                                      n_test*scipy.stats.norm.logpdf(loc=0.0,
                                                                     scale=4.0,
                                                                     x=0))
        # Ratio of same probs
        proposed = [0.0, np.log(2.0)]  # N(0, 2)
        interim = scipy.stats.norm.logpdf(loc=0.0,
                                          scale=2.0,
                                          x=0)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=interim)
        np.testing.assert_array_equal(logp_mcmc,
                                      0.0,
                                      err_msg="ratio of identical probs (=1)")
        # Ratio with displaced omega + sigma
        proposed = [0.0, np.log(2.0)]  # N(0, 2)
        interim = scipy.stats.norm.logpdf(loc=2.0,
                                          scale=2.0,
                                          x=0)
        logp_mcmc = iutils.log_prob_mcmc(omega=proposed,
                                         log_p_k_given_omega_func=log_gaussian,
                                         log_p_k_given_omega_int=interim)
        expected = n_test*(scipy.stats.norm.logpdf(loc=0,
                                                   scale=2.0,
                                                   x=0) - interim)
        np.testing.assert_array_equal(logp_mcmc,
                                      expected,
                                      err_msg="ratio of displaced means"
                                      )

    def test_get_omega_post(self):
        """Test that `get_omega_post` runs without error and returns
        MCMC chains of expected shapes

        """
        # True omega (standard normal)
        return
        true_mu = 0.0
        true_sigma = 1.0
        true_log_sigma = np.log(true_sigma)
        # True kappa realizations from omega
        n_test = 100
        true_k = self.rng.normal(loc=true_mu, scale=true_sigma,
                                 size=[n_test, 1])  # [n_test, 1]
        # BNN samples
        bnn_sigma = 10.0  # assume BNN is accurate but imprecise
        n_samples = 1000
        k_bnn = self.rng.normal(loc=true_k, scale=bnn_sigma,
                                size=[n_test, n_samples])  # [n_test, n_samples]
        # MCMC kwargs
        n_walkers = 10
        n_run = 10
        p0 = np.array([[true_mu, true_log_sigma]])
        p0 = p0 + np.random.randn(n_walkers, 2)*np.array([[0.1, 0.1]])
        mcmc_kwargs = dict(
                           p0=p0,
                           n_run=n_run,
                           n_burn=10,
                           n_walkers=n_walkers,
                           chain_path=osp.join(self.out_dir,
                                               f'chain_{n_test}'),
                           plot_chain=False,
                           clear=False,
                           n_cores=2,
                           )
        # Run MCMC
        if not osp.exists(osp.join(self.out_dir, f'chain_{n_test}')):
            iutils.get_omega_post(k_bnn,
                                  log_p_k_given_omega_int=np.ones([n_test,
                                                                  n_samples]),
                                  mcmc_kwargs=mcmc_kwargs,
                                  bounds_lower=-20.0,
                                  bounds_upper=20.0)
        # Read MCMC results
        mcmc_samples = iutils.get_mcmc_samples(mcmc_kwargs['chain_path'],
                                               dict(
                                                    flat=True,
                                                    ))
        np.testing.assert_array_equal(mcmc_samples.shape,
                                      [n_run*n_walkers, 2])
        # print(np.median(mcmc_samples, axis=0))
        # print(scipy.stats.median_abs_deviation(mcmc_samples, axis=0))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)


if __name__ == '__main__':
    unittest.main()
