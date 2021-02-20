"""Script to raytrace through cosmoD2 sightlines

Example
-------
To run this script, pass in the destination directory as the argument::

    $ python n2j/run_raytracing.py <dest_dir>

"""

import argparse
from n2j.trainval_data.raytracers.cosmodc2_raytracer import CosmoDC2Raytracer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir',
                        help='destination folder for the kappa maps, samples')
    parser.add_argument('--fov', default=0.85, dest='fov', type=float,
                        help='field of view in arcmin (Default: 0.85)')
    parser.add_argument('--n_sightlines', default=1000, dest='n_sightlines', type=int,
                        help='number of sightlines to raytrace through (Default: 1000)')
    parser.add_argument('--mass_cut', default=11.0, dest='mass_cut', type=float,
                        help='log10(minimum halo mass/solar) (Default: 11.0)')
    parser.add_argument('--n_kappa_samples', default=1000,
                        dest='n_kappa_samples', type=int,
                        help='number of kappa samples (Default: 1000)')
    args = parser.parse_args()

    sightlines_obj = CosmoDC2Raytracer(**vars(args))
    sightlines_obj.parallel_raytrace()
    sightlines_obj.apply_calibration()
