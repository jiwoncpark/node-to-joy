==================================
Hierarchical Inference with N2JNet
==================================

Once our `N2JNet` is trained, we want to generate convergence predictions on individual sightlines and combine those predictions into a hierarchical inference of the population's convergence statistics. We proceed similarly as with training, e.g.

::

$python infer.py nersc_config.yml


The "inference" section of the `nersc_config.yml` config file in the repo provides an example of how to configure inference. We take a look at it here.

First, we configure the properties of `InferenceManager`, a class that manages inference, with the device type, output directory (where all the inference results will be stored), and sampling seed.

::

    inference_manager:
      device_type: 'cpu'
      out_dir: '/global/cscratch1/sd/jwp/n2j/apj_v4/seed1/inference_E10_N1000'
      seed: 1025


Then we specify the test healpixes as well as the subsampling distribution for the test sets. For instance, if we had `n_subsample_test` of 1000 and `dist_name` of `'norm'` with `dist_kwargs` such that `loc` and `scale` were 0.04 and 0.005, respectively, we subsample 1,000 sightlines with a Gaussian distribution between with mean 0.04 and standard deviation 0.005. You can use any distribution supported by `scipy.stats`. The distributional parameters are the true hyperparameters governing the test population, which our hierarchical inference scheme will attempt to retrieve.

::

    test_data:
      seed: 1
      batch_size: 1000
      test_hp: [10326, 9686]
      n_test: [50000, 50000]
      n_subsample_test: 1000
      dist_name: 'norm'
      dist_kwargs:
        loc: 0.04
        scale: 0.005


The summary statistics matching serves as a useful comparison. We provide a grid of closeness thresholds and a minimum number of matches for a threshold to be considered valid. In the case below, we choose the smallest threshold that resulted in more than 200 matches.

::

    summary_stats:
      thresholds:
        N: [0, 1, 2, 4, 8, 16, 32, 64, 128]
        N_inv_dist: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
      min_matches: 200
    # Replace with your own


The checkpoint path of the trained `N2JNet` must be passed.

::

    checkpoint_path: '/global/cscratch1/sd/jwp/n2j/apj_v4/seed1/N2JNet_epoch=118_10-25-2021_08:10.mdl'


If we want to run hierarchical inference, we set `run_mcmc` as `True`. If we want to stop with generating individual predictions, this can be set as `False`. We can configure the MCMC, such as the number of "run" iterations, number of "burn" iterations, number of walkers, and the number of CPU cores.

::

    run_mcmc: True
    extra_mcmc_kwargs:
      n_run: 50
      n_burn: 20
      n_walkers: 10
      plot_chain: True
      clear: True
      n_cores: 1

