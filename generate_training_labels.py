"""Script to generate the training labels (structure-enhanced convergence and
shear) for >1000 sightlines

Example
-------
To run this script, pass in the destination directory as the argument::

    $ python n2j/generate_training_labels.py

"""

from n2j.trainval_data.raytracers.cosmodc2_raytracer import CosmoDC2Raytracer

if __name__ == '__main__':
    # Explicitly sample kappas for ~1000 sightlines first (slow)
    if False:
        kappa_sampler = CosmoDC2Raytracer(out_dir='kappa_sampling',
                                          fov=0.85,
                                          healpix=10450,
                                          n_sightlines=1000,  # keep this small
                                          mass_cut=11.0,
                                          n_kappa_samples=1000)
        kappa_sampler.parallel_raytrace()
        kappa_sampler.apply_calibration()
    # Use this to infer the mean kappa contribution of new sightlines
    if True:
        healpix = 10327
        train_Y_generator = CosmoDC2Raytracer(out_dir='cosmodc2_raytracing_{:d}'.format(healpix),
                                              fov=0.85,
                                              healpix=healpix,
                                              n_sightlines=20000,  # many more LOS
                                              mass_cut=11.0,
                                              n_kappa_samples=0)  # no sampling
        train_Y_generator.parallel_raytrace()
        train_Y_generator.apply_calibration()
    if True:
        healpix = 10450
        train_Y_generator = CosmoDC2Raytracer(out_dir='cosmodc2_raytracing_{:d}'.format(healpix),
                                              fov=0.85,
                                              healpix=healpix,
                                              n_sightlines=20000,  # many more LOS
                                              mass_cut=11.0,
                                              n_kappa_samples=0)  # no sampling
        train_Y_generator.parallel_raytrace()
        train_Y_generator.apply_calibration()
    if True:
        healpix = 9559
        # Use on a different healpix
        train_Y_generator = CosmoDC2Raytracer(out_dir='cosmodc2_raytracing_9559',
                                              fov=0.85,
                                              healpix=healpix,
                                              n_sightlines=1000,  # many more LOS
                                              mass_cut=11.0,
                                              n_kappa_samples=0)  # no sampling
        train_Y_generator.parallel_raytrace()
        train_Y_generator.apply_calibration()
