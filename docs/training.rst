===============
Training N2JNet
===============

To train `N2JNet`, pass in the config yml file as the argument, e.g.

::

$python train_n2jnet.py nersc_config.yml


The "training" section of the `nersc_config.yml` config file in the repo provides an example of how to configure training. We take a look at it here.

First, we configure the dataloader. This means setting the training healpixes of CosmoDC2, batch size, number of CPU cores, input galaxy features, as well as the galaxy selection and photometric noise level.

::

    # Dataloader kwargs
    data:
      train_dist_name: 'norm'
      train_dist_kwargs:
        loc: 0.01
        scale: 0.03
      in_dir: '/global/cscratch1/sd/jwp/n2j/data'
      train_hp: [10327]
      val_hp: [10450]
      n_train: [20000]
      # Final effective training set size
      n_subsample_train: 50000
      n_val: [50000]
      # Final effective val set size
      n_subsample_val: 1000
      batch_size: 1000
      val_batch_size: 1000
      num_workers: 8
      # Global (graph-level) target; final_gamma1, final_gamm2 also available
      sub_target: ['final_kappa']
      # Local (node-level) target
      sub_target_local: ['stellar_mass', 'redshift']
      # Features available; do not modify (determined at data generation time)
      features: ['galaxy_id', 'ra', 'dec', 'redshift',
                  'ra_true', 'dec_true', 'redshift_true',
                  'bulge_to_total_ratio_i',
                  'ellipticity_1_true', 'ellipticity_2_true',
                  'ellipticity_1_bulge_true', 'ellipticity_1_disk_true',
                  'ellipticity_2_bulge_true', 'ellipticity_2_disk_true',
                  'shear1', 'shear2', 'convergence',
                  'size_bulge_true', 'size_disk_true', 'size_true',
                  'mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst',
                  'mag_i_lsst', 'mag_z_lsst', 'mag_Y_lsst']
      # Features to use as input
      sub_features: ['ra_true', 'dec_true',
                      'mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst',
                      'mag_i_lsst', 'mag_z_lsst', 'mag_Y_lsst']
      noise_kwargs:
        mag:
          override_kwargs: null
          depth: 5
      detection_kwargs:
        ref_features: ['mag_i_lsst']
        max_vals: [22.0]


Then we configure the optimizer, i.e. the weighting between the loss of global convergence labels and local stellar mass and redshift labels, early-stopping, initial learning rate, learning rate decay.

::

    # Optimizer kwargs
    optimization:
      early_stop_memory: 50
      weight_local_loss: 1.0
      optim_kwargs:
        lr: 0.001
        weight_decay: 0.0001
      lr_scheduler_kwargs:
        patience: 5
        factor: 0.5
        min_lr: 0.0000001
        verbose: True


We then configure the depth and width of the network architecture. The `global flow` key determines whether the final layer predicting the convergence is a normalizing flow.

::

    # Model kwargs
    model:
      dim_local: 50
      dim_global: 50
      dim_hidden: 50
      dim_pre_aggr: 50
      n_iter: 5
      n_out_layers: 5
      dropout: 0.04
      global_flow: False


Lastly, we set more general attributes of training such as the seed, device, and the number of training epochs.

::

    # Trainer attributes
    trainer:
      device_type: 'cuda'
      checkpoint_dir: results/E1
      seed: 1028
    n_epochs: 200
    # If you want to resume training from a checkpoint
    resume_from:
      checkpoint_path: null
