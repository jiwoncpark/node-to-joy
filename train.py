"""Script to train the Bayesian GNN

"""
import sys
import cProfile
from n2j.trainval_data.raytracers.cosmodc2_raytracer import CosmoDC2Raytracer
from n2j.trainer import Trainer

if __name__ == '__main__':
    IN_DIR = '/home/jwp/stage/sl/n2j/n2j/data'  # where raw data lies
    TRAIN_HP = [10327, 10450]
    VAL_HP = [9559]
    N_TRAIN = 10
    N_VAL = 10
    CHECKPOINT_PATH = None

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
    # Use this to infer the mean kappa contribution of new sightlines
    for hp in TRAIN_HP:
        train_Y_generator = CosmoDC2Raytracer(in_dir=IN_DIR,
                                              out_dir=f'Y_{hp}',
                                              fov=0.85,
                                              healpix=hp,
                                              n_sightlines=N_TRAIN,  # many more LOS
                                              mass_cut=11.0,
                                              n_kappa_samples=0)  # no sampling
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
                                            n_kappa_samples=0)  # no sampling
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
    sub_features += ['size_true']
    sub_features += ['mag_{:s}_lsst'.format(b) for b in 'i']
    trainer = Trainer('cuda', checkpoint_dir='test_run', seed=1234)

    trainer.load_dataset(dict(features=features,
                              raytracing_out_dirs=[f'Y_{hp}' for hp in TRAIN_HP],
                              healpixes=TRAIN_HP,
                              n_data=[N_TRAIN]*len(TRAIN_HP),
                              aperture_size=1.0,
                              stop_mean_std_early=False,
                              in_dir=IN_DIR),
                         sub_features=sub_features,
                         is_train=True,
                         batch_size=2,
                         )
    # FIXME: must be run after train
    trainer.load_dataset(dict(features=features,
                              raytracing_out_dirs=[f'Y_{hp}' for hp in VAL_HP],
                              healpixes=VAL_HP,
                              n_data=[N_VAL]*len(VAL_HP),
                              aperture_size=1.0,
                              in_dir=IN_DIR),
                         sub_features=sub_features,
                         is_train=False,
                         batch_size=2,  # FIXME: must be same as train
                         )
    trainer.configure_loss_fn('FullRankGaussianNLL')
    trainer.configure_model('GATNet',
                            {'hidden_channels': 16,
                             'n_layers': 3,
                             'dropout': 0.0,
                             'kwargs': {'concat': False, 'heads': 4}})
    trainer.configure_optim(20,
                            {'lr': 1.e-3, 'weight_decay': 1.e-5},
                            {'factor': 0.5, 'min_lr': 1.e-7, 'patience': 20, 'verbose': True})
    if CHECKPOINT_PATH:
        trainer.load_state(CHECKPOINT_PATH)
    trainer.train(n_epochs=5)
    print(trainer)
