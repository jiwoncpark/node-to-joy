"""Script to raytrace through cosmoD2 sightlines

Example
-------
To run this script, pass in the destination directory as the argument::
    
    $ python extranet/raytrace_cosmodc2.py <dest_dir>

"""

import os
import sys
import itertools
import argparse
import functools
import pandas as pd
from tqdm import tqdm
import multiprocessing
from extranet.trainval_data.raytracing_utils import (raytrace_single_sightline, 
get_sightlines_random)

def single_raytrace(i, healpix, sightlines, fov, map_kappa, map_gamma,
                    n_kappa_samples, mass_cut, dest_dir):
    """Wrapper around `raytrace_single_sightline` to enable multiprocessing"""
    sightline = sightlines.iloc[i]
    raytrace_single_sightline(i, 
                              healpix,
                              sightline['ra'], sightline['dec'],
                              sightline['gal_z'], 
                              fov,
                              map_kappa,
                              map_gamma,
                              n_kappa_samples,
                              mass_cut,
                              dest_dir)
    return None

class Sightlines:
    """Set of sightlines in a cosmoDC2 field"""
    def __init__(self, dest_dir, fov, map_kappa, map_gamma, one_sightline,
                 mass_cut=11, n_sightlines=1000):
        """
        Parameters
        ----------
        dest_dir : str or os.path
        fov : float
            field of view in arcmin
        map_kappa : bool
            whether to generate grid maps of kappa
        mass_cut : float
            log10(minimum halo mass) (Default: 11.0)
        n_sightlines : int
            number of sightlines to raytrace through (Default: 1000)

        """
        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.mkdir(self.dest_dir)
        self.fov = fov
        self.map_kappa = map_kappa
        self.map_gamma = map_gamma
        self.one_sightline = one_sightline # FIXME
        self.mass_cut = mass_cut
        self.n_sightlines = n_sightlines
        self.healpix = 10450
        self._get_pointings()
        self.uncalib_path = os.path.join(self.dest_dir, 'uncalib.txt')
        open(self.uncalib_path, 'a').close()
        
    def _get_pointings(self):
        sightlines_path = '{:s}/random_sightlines.csv'.format(self.dest_dir)
        if os.path.exists(sightlines_path):
            self.pointings = pd.read_csv(sightlines_path, 
                                         index_col=None,
                                         nrows=self.n_sightlines)
        else:
            self.pointings = get_sightlines_random(self.healpix,
                                                   self.n_sightlines, 
                                                   sightlines_path,
                                                   edge_buffer=self.fov*0.5)

    def parallel_raytrace(self):
        single = functools.partial(single_raytrace, 
                                   healpix=self.healpix,
                                   sightlines=self.pointings, 
                                   fov=self.fov, 
                                   map_kappa=self.map_kappa, 
                                   map_gamma=self.map_gamma,
                                   n_kappa_samples=1000,
                                   mass_cut=self.mass_cut,
                                   dest_dir=self.dest_dir)
        #return pool.map(single, )
        return list(tqdm(pool.imap(single, range(self.n_sightlines)), 
                         total=self.n_sightlines))

if __name__ == '__main__':
    #get_sightlines()
    #
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir',
                        help='destination folder for the kappa maps, samples')
    parser.add_argument('--fov', default=6.0, dest='fov', type=float,
                        help='field of view in arcmin (Default: 6.0)')
    parser.add_argument('--map_kappa', default=False, dest='map_kappa', type=bool,
                        help='whether to generate grid maps of kappa (Default: False)')
    parser.add_argument('--map_gamma', default=False, dest='map_gamma', type=bool,
                        help='whether to generate grid maps of gamma (Default: False)')
    parser.add_argument('--n_sightlines', default=1000, dest='n_sightlines', type=int,
                        help='number of sightlines to raytrace through (Default: 1000)')
    parser.add_argument('--mass_cut', default=11.0, dest='mass_cut', type=float,
                        help='log10(minimum halo mass/solar) (Default: 11.0)')
    parser.add_argument('--one_sightline', default=None, dest='one_sightline', type=int,
                        help='index of a single sightline to run')
    args = parser.parse_args()

    n_cores = min(multiprocessing.cpu_count() - 1, args.n_sightlines)
    sightlines = Sightlines(**vars(args))
    with multiprocessing.Pool(n_cores) as pool:
        sightlines.parallel_raytrace()
    #pr.disable()
    #pr.print_stats(sort='cumtime')
    #for i in tqdm(range(n_sightlines), desc="Raytracing through each sightline"):
    #    
    #raytrace(fov=6.0, map_kappa=True, n_sightlines=1, n_kappa_samples=5)
    #
    #
    #

#self, z_source, lens_model_list, lens_redshift_list, cosmo=None, numerical_alpha_class=None, observed_convention_index=None, ignore_observed_positions=False, z_source_convention=None

#mag_g_lsst,baseDC2/target_halo_z,ellipticity_1_true,size_minor_disk_true,baseDC2/host_halo_vx,mag_z_lsst,shear1,baseDC2/target_halo_vx,baseDC2/host_halo_x,shear_2_phosim,mag_u_lsst,mag_i_lsst,baseDC2/host_halo_vy,baseDC2/host_halo_z,redshift_true,
#baseDC2/target_halo_redshift,baseDC2/host_halo_vz,baseDC2/target_halo_vz,baseDC2/target_halo_vy,mag_Y_lsst,dec,convergence,baseDC2/target_halo_fof_halo_id,baseDC2/target_halo_mass,ellipticity_bulge_true,baseDC2/halo_id,shear_1,baseDC2/target_halo_id,shear2,baseDC2/host_halo_y,ellipticity_2_bulge_true,size_minor_true,galaxy_id,ellipticity_2_disk_true,stellar_mass,position_angle_true,baseDC2/target_halo_x,baseDC2/target_halo_y,ellipticity_2_true,size_true,ellipticity_1_bulge_true,halo_mass,mag_r_lsst,baseDC2/source_halo_id,baseDC2/source_halo_mvir,halo_id,size_disk_true,shear_2,bulge_to_total_ratio_i,size_minor_bulge_true,baseDC2/host_halo_mvir,size_bulge_true,ellipticity_1_disk_true,stellar_mass_bulge,ra,stellar_mass_disk,ellipticity_disk_true,ellipticity_true,shear_2_treecorr,redshift

