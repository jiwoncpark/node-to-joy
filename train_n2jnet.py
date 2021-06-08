"""Script to train the Bayesian GNN

"""
import os
import sys
import cProfile
import numpy as np
from n2j.trainval_data.raytracers.cosmodc2_raytracer import CosmoDC2Raytracer
from n2j.trainer import Trainer

if __name__ == '__main__':
    IN_DIR = '/home/jwp/stage/sl/n2j/n2j/data'  # where raw data lies
    TRAIN_HP = [10450, 10327, 10200]
    VAL_HP = [10326]
    N_TRAIN = 20000
    N_VAL = 100
    BATCH_SIZE = 1000  # min(N_TRAIN//5, 50)
    CHECKPOINT_PATH = None
    SUB_TARGET = ['final_kappa', ]  # 'final_gamma1', 'final_gamma2']
    SUB_TARGET_LOCAL = ['redshift']
    CHECKPOINT_DIR = 'results/E0'
    SKIP_RAYTRACING = True

    ##############
    # Labels (Y) #
    ##############
    # Explicitly sample kappas for ~1000 sightlines first (slow)
    if False:
        kappa_sampler = CosmoDC2Raytracer(in_dir=IN_DIR,
                                          out_dir='kappa_sampling',
                                          fov=0.85,
                                          healpix=10450,
                                          n_sightlines=1000,  # keep this small
                                          mass_cut=11.0,
                                          n_kappa_samples=1000)
        kappa_sampler.parallel_raytrace()
        kappa_sampler.apply_calibration()
    if not SKIP_RAYTRACING:
        # Use this to infer the mean kappa contribution of new sightlines
        for hp in TRAIN_HP:
            train_Y_generator = CosmoDC2Raytracer(in_dir=IN_DIR,
                                                  out_dir=f'Y_{hp}',
                                                  fov=0.85,
                                                  healpix=hp,
                                                  n_sightlines=N_TRAIN,  # many more LOS
                                                  mass_cut=11.0,
                                                  n_kappa_samples=0,
                                                  kappa_sampling_dir='kappa_sampling')  # no sampling
            train_Y_generator.parallel_raytrace()
            train_Y_generator.apply_calibration()

        for hp in VAL_HP:
            # Use on a different healpix
            val_Y_generator = CosmoDC2Raytracer(in_dir=IN_DIR,
                                                out_dir=f'Y_{hp}',
                                                fov=0.85,
                                                healpix=hp,
                                                n_sightlines=N_VAL,  # many more LOS
                                                mass_cut=11.0,
                                                n_kappa_samples=0,
                                                kappa_sampling_dir='kappa_sampling')  # no sampling
            val_Y_generator.parallel_raytrace()
            val_Y_generator.apply_calibration()

    ##############
    # Graphs (X) #
    ##############

    features = ['galaxy_id', 'ra', 'dec', 'redshift']
    features += ['ra_true', 'dec_true', 'redshift_true']
    features += ['ellipticity_1_true', 'ellipticity_2_true']
    features += ['bulge_to_total_ratio_i']
    features += ['ellipticity_1_bulge_true', 'ellipticity_1_disk_true']
    features += ['ellipticity_2_bulge_true', 'ellipticity_2_disk_true']
    features += ['shear1', 'shear2', 'convergence']
    features += ['size_bulge_true', 'size_disk_true', 'size_true']
    features += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']
    # Features to train on
    sub_features = ['ra_true', 'dec_true']
    # sub_features += ['size_true']
    # sub_features += ['ellipticity_1_true', 'ellipticity_2_true']
    sub_features += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']
    trainer = Trainer('cuda', checkpoint_dir=CHECKPOINT_DIR, seed=1028)

    trainer.load_dataset(dict(features=features,
                              raytracing_out_dirs=[f'Y_{hp}' for hp in TRAIN_HP],
                              healpixes=TRAIN_HP,
                              n_data=[N_TRAIN]*len(TRAIN_HP),
                              aperture_size=1.0,
                              stop_mean_std_early=False,
                              in_dir=IN_DIR),
                         sub_features=sub_features,
                         sub_target=SUB_TARGET,
                         sub_target_local=SUB_TARGET_LOCAL,
                         is_train=True,
                         batch_size=BATCH_SIZE,
                         )

    # FIXME: must be run after train
    trainer.load_dataset(dict(features=features,
                              raytracing_out_dirs=[f'Y_{hp}' for hp in VAL_HP],
                              healpixes=VAL_HP,
                              n_data=[N_VAL]*len(VAL_HP),
                              aperture_size=1.0,
                              in_dir=IN_DIR),
                         sub_features=sub_features,
                         sub_target=SUB_TARGET,
                         sub_target_local=SUB_TARGET_LOCAL,
                         is_train=False,
                         batch_size=N_VAL,  # FIXME: must be same as train
                         )

    print(trainer.Y_local_mean, trainer.Y_local_std)
    print(trainer.Y_mean, trainer.Y_std)
    if False:
        print(trainer.train_dataset[0].y_local)
        for b in trainer.train_loader:
            print(b.x.shape, b.y_local.shape, b.y.shape, b.batch.shape)
            print(b.y_local[:5, 0])
            break

    trainer.configure_loss_fn('MSELoss')
    model_kwargs = dict(dim_in=trainer.X_dim,
                        dim_out_local=len(SUB_TARGET_LOCAL),
                        dim_out_global=len(SUB_TARGET),
                        dim_local=50,
                        dim_global=50,
                        dim_hidden=50,
                        dim_pre_aggr=50,
                        n_iter=5,
                        n_out_layers=5,
                        global_flow=False
                        )
    trainer.configure_model('N2JNet', model_kwargs)

    trainer.configure_optim(early_stop_memory=50,
                            weight_local_loss=0.01,
                            optim_kwargs={'lr': 1e-3, 'weight_decay': 1.e-5},
                            lr_scheduler_kwargs={'factor': 0.5, 'min_lr': 1.e-7, 'patience': 5, 'verbose': True})
    if CHECKPOINT_PATH:
        trainer.load_state(CHECKPOINT_PATH)
    trainer.train(n_epochs=200, eval_every=2)
    sys.exit()
    print(trainer)
    # Save final validation metrics
    summary = trainer.eval_posterior(epoch_i=trainer.epoch,
                                     n_samples=200,
                                     n_mc_dropout=20,
                                     on_train=False)
    np.save(os.path.join(trainer.checkpoint_dir, 'summary.npy'), summary)
