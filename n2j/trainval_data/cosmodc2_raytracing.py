"""This module contains utility functions for raytracing through cosmoD2 halos.

"""

import os
import time
import yaml
from astropy.cosmology import WMAP7   # WMAP 7-year cosmology
import numpy as np
import pandas as pd
# import healpy as hp
from lenstronomy.LensModel.lens_model import LensModel
import n2j.trainval_data.coord_utils as cu
import n2j.trainval_data.halo_utils as hu
from n2j import trainval_data, data

__all__ = ['get_cosmodc2_generator']
__all__ += ['get_sightlines_on_grid']
__all__ += ['get_los_halos']
__all__ += ['raytrace_single_sightline']

KAPPA_DIFF = 1.0
meta_path = os.path.join(trainval_data.__path__[0], 'catalog_metadata.yaml')
with open(meta_path) as file:
    meta = yaml.load(file, Loader=yaml.FullLoader)
    NSIDE = meta['cosmodc2']['nside']
    # lensing_nside = meta['cosmodc2']['lensing_nside']
# FOV = hp.nside2resol(lensing_nside, arcmin=True)


def get_cosmodc2_generator(healpix, columns=None, chunksize=100000, small=False):
    """Get a generator of cosmoDC2, too big to store in memory at once

    Parameters
    ----------
    columns : list
        list of columns to load. Must match the CSV header.
    chunksize : int
        number of rows in each chunk
    small : bool
        whether to load a small CSV of only 1000 rows, for testing purposes.

    """
    if small:
        cosmodc2_path = os.path.join(data.__path__[0], 'test_data.csv')
    else:
        cosmodc2_path = os.path.join(data.__path__[0],
                                     'cosmodc2_train', 'raw',
                                     'cosmodc2_trainval_{:d}.csv'.format(healpix))
    cosmodc2 = pd.read_csv(cosmodc2_path, chunksize=chunksize, nrows=None,
                           usecols=columns)
    return cosmodc2


def rename_cosmodc2_cols(df):
    """Rename cosmoDC2-specific columns to more general ones

    """
    column_names = meta['cosmodc2']['column_names']
    df.rename(columns=column_names, inplace=True)


def get_sightlines_on_grid(healpix, n_sightlines, out_path,
                           dist_thres, test=False):
    """Get the sightlines

    Parameters
    ----------
    healpix : int
        healpix ID that will be supersampled
    n_sightlines : int
        desired number of sightlines
    dist_thres : float
        matching threshold between gridpoints and halo positions, in deg
    out_path : str or os.path instance
        where the output file `sightlines.csv` will be stored

    Notes
    -----
    Currently takes 1.2 min for 1000 sightlines.
    Doesn't have to be so rigorous about finding sightlines closest to grid.
    Two requirements are that sightlines need to be dominated by cosmic variance
    (span a few degrees) and that each sightline has a galaxy.

    """

    # Get centroids of D partitions by gridding the sky area and querying a
    # galaxy closest to each grid center at redshift z > 2
    # Each partition, centered at that galaxy,
    # corresponds to a line of sight (LOS)
    start = time.time()
    target_nside = cu.get_target_nside(n_sightlines,
                                       nside_in=NSIDE)
    sightline_ids = cu.upgrade_healpix(healpix, False,
                                       NSIDE, target_nside)
    ra_grid, dec_grid = cu.get_healpix_centers(sightline_ids, target_nside,
                                               nest=True)
    # Randomly choose number of sightlines requested
    rand_i = np.random.choice(np.arange(len(ra_grid)), size=n_sightlines,
                              replace=False)
    ra_grid, dec_grid = ra_grid[rand_i], dec_grid[rand_i]
    close_enough = np.zeros_like(ra_grid).astype(bool)  # all gridpoints False
    sightline_cols = ['ra_true', 'dec_true', 'redshift_true', 'galaxy_id']
    sightline_cols += ['convergence', 'shear1', 'shear2']
    cosmodc2 = get_cosmodc2_generator(healpix, sightline_cols, small=test)
    sightlines = pd.DataFrame()
    for df in cosmodc2:
        high_z = df[(df['redshift_true'] > 2.0)].reset_index(drop=True)
        if len(high_z) > 0:
            remaining = ~close_enough
            passing, i_cat, dist = cu.match(ra_grid[remaining],
                                            dec_grid[remaining],
                                            high_z['ra_true'].values,
                                            high_z['dec_true'].values,
                                            dist_thres
                                            )
            more_sightlines = high_z.iloc[i_cat].copy()
            more_sightlines['eps'] = dist
            sightlines = sightlines.append(more_sightlines, ignore_index=True)
            close_enough[remaining] = passing
        if np.all(close_enough):
            break
    sightlines.reset_index(drop=True, inplace=True)
    rename_cosmodc2_cols(sightlines)
    sightlines.to_csv(out_path, index=None)
    end = time.time()
    print("Generated {:d} sightline(s) in"
          " {:.2f} min.".format(n_sightlines, (end-start)/60.0))
    return sightlines


def get_los_halos(generator, ra_los, dec_los, z_src, fov, mass_cut, out_path):
    halos = pd.DataFrame()  # neighboring galaxies in LOS
    # Iterate through chunks to bin galaxies into the partitions
    for df in generator:
        # Get galaxies in the aperture and in foreground of source
        # Discard smaller masses, since they won't have a big impact anyway
        lower_z = df['redshift_true'].values < z_src
        if lower_z.any():  # there are still some lower-z halos
            pass
        else:  # z started getting too high, no need to continue
            continue
        high_mass = df['halo_mass'].values > 10.0**mass_cut
        central_only = (df['is_central'].values == True)
        cut = np.logical_and(np.logical_and(high_mass, lower_z), central_only)
        df = df[cut].reset_index(drop=True)
        if len(df) > 0:
            d, ra_diff, dec_diff = cu.get_distance(ra_f=df['ra_true'].values,
                                                   dec_f=df['dec_true'].values,
                                                   ra_i=ra_los,
                                                   dec_i=dec_los
                                                   )
            df['dist'] = d*60.0  # deg to arcmin
            df['ra_diff'] = ra_diff  # deg
            df['dec_diff'] = dec_diff  # deg
            halos = halos.append(df[df['dist'].values < fov*0.5],
                                 ignore_index=True)
        else:
            continue

    #####################
    # Define NFW kwargs #
    #####################
    halos['center_x'] = halos['ra_diff']*3600.0  # deg to arcsec
    halos['center_y'] = halos['dec_diff']*3600.0
    Rs, alpha_Rs, eff = hu.get_nfw_kwargs(halos['halo_mass'].values,
                                          halos['stellar_mass'].values,
                                          halos['redshift_true'].values,
                                          z_src)
    halos['Rs'] = Rs
    halos['alpha_Rs'] = alpha_Rs
    halos['eff'] = eff
    halos.reset_index(drop=True, inplace=True)
    rename_cosmodc2_cols(halos)
    halos.to_csv(out_path, index=None)
    return halos


def raytrace_single_sightline(idx, healpix, ra_los, dec_los, z_src, fov,
                              map_kappa, map_gamma,
                              n_kappa_samples, mass_cut, dest_dir,
                              test=False):
    """Raytrace through a single sightline

    """
    halo_filename = '{:s}/los_halos_los={:d}.csv'.format(dest_dir, idx)
    if os.path.exists(halo_filename):
        halos = pd.read_csv(halo_filename, index_col=None)
    else:
        halo_cols = ['halo_mass', 'stellar_mass', 'is_central']
        halo_cols += ['ra_true', 'dec_true', 'redshift_true']
        cosmodc2 = get_cosmodc2_generator(healpix, halo_cols, small=test)
        halos = get_los_halos(cosmodc2, ra_los, dec_los, z_src, fov, mass_cut, halo_filename)
    n_halos = halos.shape[0]
    # Instantiate multi-plane lens model
    lens_model = LensModel(lens_model_list=['NFW']*n_halos,
                           z_source=z_src,
                           lens_redshift_list=halos['z'].values,
                           multi_plane=True,
                           cosmo=WMAP7,
                           observed_convention_index=[])
    nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
    uncalib_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=KAPPA_DIFF,
                                     diff_method='square')
    uncalib_gamma1, uncalib_gamma2 = lens_model.gamma(0.0, 0.0, nfw_kwargs,
                                                      diff=KAPPA_DIFF,
                                                      diff_method='square')
    # Log the uncalibrated shear and convergence
    new_row_data = dict(idx=[idx],
                        kappa=[uncalib_kappa],
                        gamma1=[uncalib_gamma1],
                        gamma2=[uncalib_gamma2],
                        )
    uncalib_path = os.path.join(dest_dir, 'uncalib.csv')  # FIXME
    new_row = pd.DataFrame(new_row_data)
    new_row.to_csv(uncalib_path, index=None, mode='a', header=None)
    # Optionally map the uncalibrated shear and convergence on a grid
    if map_kappa:
        hu.get_kappa_map(lens_model, nfw_kwargs, fov,
                         '{:s}/k_map_los={:d}.npy'.format(dest_dir, idx))
        if map_gamma:
            hu.get_gamma_maps(lens_model, nfw_kwargs, fov,
                              ('{:s}/g1_map_los={:d}.npy'.format(dest_dir, idx),
                               '{:s}/g2_map_los={:d}.npy'.format(dest_dir, idx)))

    ################
    # Sample kappa #
    ################
    # gamma1, gamma2 are not resampled due to symmetry around 0
    kappa_samples_path = '{:s}/k_samples_los={:d}.npy'.format(dest_dir, idx)
    if os.path.exists(kappa_samples_path):
        pass
    else:
        kappa_samples = np.empty(n_kappa_samples)
        S = 0
        while S < n_kappa_samples:
            new_ra, new_dec = cu.sample_in_aperture(n_halos, fov*0.5/60.0)
            halos['center_x'] = new_ra*3600.0
            halos['center_y'] = new_dec*3600.0
            nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
            resampled_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=KAPPA_DIFF,
                                               diff_method='square')
            if resampled_kappa < 1.0:
                kappa_samples[S] = resampled_kappa
                if map_kappa and test:
                    hu.get_kappa_map(lens_model, nfw_kwargs, fov,
                                     '{:s}/k_map_los={:d}_sample={:d}.npy'.format(dest_dir, idx, S))
                S += 1
            else:  # halo fell on top of zeropoint!
                continue
        np.save(kappa_samples_path, kappa_samples)
