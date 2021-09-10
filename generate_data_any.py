"""Script to generate training data

"""
import os
import argparse
from scipy import stats
from n2j.trainval_data.raytracers.cosmodc2_raytracer import CosmoDC2Raytracer
from n2j.trainer import Trainer
from n2j.config_utils import get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('healpix_id',
                        help='healpix ID of CosmoDC2', type=int)
    args = parser.parse_args()
    IN_DIR = '/global/cscratch1/sd/jwp/n2j/data'  # where raw data lies
    TRAIN_HP = [args.healpix_id] #, 10198, 10072, 10071, 10070, 9943, 9942, 9816, 9815, 9814, 9687, 9686, 9559, 10450, 10327, 10326, 10200]
    N_TRAIN = 50000
    BATCH_SIZE = 1000  # min(N_TRAIN//5, 50)
    CHECKPOINT_PATH = None
    SUB_TARGET = ['final_kappa', ]  # 'final_gamma1', 'final_gamma2']
    SUB_TARGET_LOCAL = ['redshift']
    CHECKPOINT_DIR = f'dummy/{args.healpix_id}'

    ##############
    # Labels (Y) #
    ##############
    # Explicitly sample kappas for ~1000 sightlines first (slow)
    if False:
        kappa_sampler = CosmoDC2Raytracer(in_dir=IN_DIR,
                                          out_dir=f'/global/cscratch1/sd/jwp/n2j/data/kappa_sampling',
                                          fov=1.35,
                                          healpix=10450,
                                          n_sightlines=1000,  # keep this small
                                          mass_cut=11.0,
                                          n_kappa_samples=1000,
                                          seed=123)
        kappa_sampler.parallel_raytrace(n_cores=200)
        kappa_sampler.apply_calibration()
    # Use this to infer the mean kappa contribution of new sightlines
    for hp in TRAIN_HP:
        print(f"Raytracing for healpix {hp}...")
        train_Y_generator = CosmoDC2Raytracer(in_dir=IN_DIR,
                                              out_dir=f'/global/cscratch1/sd/jwp/n2j/data/cosmodc2_{hp}/Y_{hp}',
                                              fov=1.35,
                                              healpix=hp,
                                              n_sightlines=N_TRAIN,  # many more LOS
                                              mass_cut=11.0,
                                              n_kappa_samples=0,
                                              seed=hp,
                                              kappa_sampling_dir=f'/global/cscratch1/sd/jwp/n2j/data/kappa_sampling')  # no sampling
        train_Y_generator.parallel_raytrace(n_cores=200)
        train_Y_generator.apply_calibration()

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
    norm_obj = getattr(stats, 'norm')(loc=0.01, scale=0.04)
    trainer.load_dataset(dict(features=features,
                              raytracing_out_dirs=[f'/global/cscratch1/sd/jwp/n2j/data/cosmodc2_{hp}/Y_{hp}' for hp in TRAIN_HP],
                              healpixes=TRAIN_HP,
                              n_data=[N_TRAIN]*len(TRAIN_HP),
                              aperture_size=1.0,
                              subsample_pdf_func=norm_obj.pdf,
                              n_subsample=2000,
                              stop_mean_std_early=False,
                              in_dir=IN_DIR,
                              n_cores=200),
                         sub_features=sub_features,
                         sub_target=SUB_TARGET,
                         sub_target_local=SUB_TARGET_LOCAL,
                         is_train=True,
                         batch_size=BATCH_SIZE,
                         num_workers=20,
                         rebin=False,
                         noise_kwargs=dict(mag=dict(override_kwargs=None,
                                                    depth=5)),
                         detection_kwargs={},
                         )
