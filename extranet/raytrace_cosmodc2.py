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
from extranet.trainval_data.raytracing_utils import raytrace_single_sightline

def single_raytrace(i, sightlines, fov, map_kappa, n_kappa_samples, dest_dir):
    sightline = sightlines.iloc[i]
    raytrace_single_sightline(i, 
                              sightline['ra'], sightline['dec'],
                              sightline['redshift'], 
                              sightline['convergence'],
                              fov,
                              map_kappa,
                              n_kappa_samples,
                              dest_dir)
    return None

class Sightlines:
    """Set of sightlines in a cosmoDC2 field"""
    def __init__(self, dest_dir):
        """
        Parameters
        ----------
        dest_dir : str or os.path

        """
        sightlines_path = '{:s}/random_sightlines.csv'.format(dest_dir)
        self.sightlines = pd.read_csv(sightlines_path, index_col=None)
        self.fov = 6.0
        self.map_kappa = False
        self.n_sightlines = self.sightlines.shape[0]
        self.dest_dir = dest_dir

    def parallel_raytrace(self):
        single = functools.partial(single_raytrace, 
                                   sightlines=self.sightlines, 
                                   fov=self.fov, 
                                   map_kappa=self.map_kappa, 
                                   n_kappa_samples=self.n_sightlines,
                                   dest_dir=self.dest_dir)
        #return pool.map(single, )
        return list(tqdm(pool.imap(single, range(self.n_sightlines)), 
                         total=self.n_sightlines))

if __name__ == '__main__':
    #get_sightlines()
    #get_sightlines_random(n_sightlines=1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir',
                        help='destination folder for the kappa maps, samples')
    args = parser.parse_args()

    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        Sightlines(args.dest_dir).parallel_raytrace()

    #for i in tqdm(range(n_sightlines), desc="Raytracing through each sightline"):
    #    
    #raytrace(fov=6.0, map_kappa=True, n_sightlines=1, n_kappa_samples=5)
    #
    #
    #

#self, z_source, lens_model_list, lens_redshift_list, cosmo=None, numerical_alpha_class=None, observed_convention_index=None, ignore_observed_positions=False, z_source_convention=None

#mag_g_lsst,baseDC2/target_halo_z,ellipticity_1_true,size_minor_disk_true,baseDC2/host_halo_vx,mag_z_lsst,shear1,baseDC2/target_halo_vx,baseDC2/host_halo_x,shear_2_phosim,mag_u_lsst,mag_i_lsst,baseDC2/host_halo_vy,baseDC2/host_halo_z,redshift_true,
#baseDC2/target_halo_redshift,baseDC2/host_halo_vz,baseDC2/target_halo_vz,baseDC2/target_halo_vy,mag_Y_lsst,dec,convergence,baseDC2/target_halo_fof_halo_id,baseDC2/target_halo_mass,ellipticity_bulge_true,baseDC2/halo_id,shear_1,baseDC2/target_halo_id,shear2,baseDC2/host_halo_y,ellipticity_2_bulge_true,size_minor_true,galaxy_id,ellipticity_2_disk_true,stellar_mass,position_angle_true,baseDC2/target_halo_x,baseDC2/target_halo_y,ellipticity_2_true,size_true,ellipticity_1_bulge_true,halo_mass,mag_r_lsst,baseDC2/source_halo_id,baseDC2/source_halo_mvir,halo_id,size_disk_true,shear_2,bulge_to_total_ratio_i,size_minor_bulge_true,baseDC2/host_halo_mvir,size_bulge_true,ellipticity_1_disk_true,stellar_mass_bulge,ra,stellar_mass_disk,ellipticity_disk_true,ellipticity_true,shear_2_treecorr,redshift

