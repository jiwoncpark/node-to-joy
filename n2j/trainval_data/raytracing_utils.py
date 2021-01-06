import yaml
import os
import time
import numpy as np
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
#from lenstronomy.LensModel.Profiles.nfw import NFW
from astropy.cosmology import WMAP7   # WMAP 7-year cosmology
import n2j.trainval_data.coord_utils as cu
from n2j import trainval_data

kappa_diff = 1.0
__all__ = ['get_cosmodc2_generator']
__all__ += ['get_sightlines_on_grid']
__all__ += ['get_los_halos', 'get_nfw_kwargs', 'get_kappa_map']
__all__ += ['get_concentration']
__all__ += ['raytrace_single_sightline']


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
    from n2j import data
    if small:
        cosmodc2_path = os.path.join(data.__path__[0], 'test_data.csv')
    else:
        cosmodc2_path = os.path.join(data.__path__[0],
                                     'cosmodc2_train', 'raw',
                                     'cosmodc2_trainval_{:d}.csv'.format(healpix))
    cosmodc2 = pd.read_csv(cosmodc2_path, chunksize=chunksize, nrows=None,
                           usecols=columns)
    meta_path = os.path.join(trainval_data.__path__[0], 'catalog_metadata.yaml')
    with open(meta_path) as file:
        meta = yaml.load(file, Loader=yaml.FullLoader)
        h = meta['cosmodc2']['cosmology']['H0']/100.0
    cosmodc2.h = h
    return cosmodc2


def rename_cosmodc2_cols(df):
    """Rename cosmoDC2-specific columns to more general ones

    """
    meta_path = os.path.join(trainval_data.__path__[0], 'catalog_metadata.yaml')
    with open(meta_path) as file:
        meta = yaml.load(file, Loader=yaml.FullLoader)
        column_names = meta['cosmodc2']['column_names']
    df.rename(columns=column_names, inplace=True)


def get_sightlines_on_grid(healpix, n_sightlines, out_path,
                           dist_thres=6.0/3600.0, test=False):
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
    target_nside = cu.get_target_nside(n_sightlines, nside_in=2**5)
    sightline_ids = cu.upgrade_healpix(healpix, False, 2**5, target_nside)
    ra_grid, dec_grid = cu.get_healpix_centers(sightline_ids, target_nside,
                                               nest=True)
    # Randomly choose number of sightlines requested
    rand_i = np.random.choice(np.arange(len(ra_grid)), size=n_sightlines,
                              replace=False)
    ra_grid, dec_grid = ra_grid[rand_i], dec_grid[rand_i]
    close_enough = np.zeros_like(ra_grid).astype(bool)  # all gridpoints False
    sightline_cols = ['ra_true', 'dec_true', 'redshift_true']
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
    print("Generated {:d} sightlines in {:.2f} min.".format(n_sightlines,
                                                            (end-start)/60.0))
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
    Rs, alpha_Rs, eff = get_nfw_kwargs(halos['halo_mass'].values,
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


def get_nfw_kwargs(halo_mass, stellar_mass, halo_z, z_src):
    c_200 = get_concentration(halo_mass, stellar_mass)
    n_halos = len(halo_mass)
    Rs_angle, alpha_Rs = np.empty(n_halos), np.empty(n_halos)
    lensing_eff = np.empty(n_halos)
    for h in range(n_halos):
        lens_cosmo = LensCosmo(z_lens=halo_z[h], z_source=z_src, cosmo=WMAP7)
        lensing_eff = lens_cosmo.dds/lens_cosmo.ds
        Rs_angle_h, alpha_Rs_h = lens_cosmo.nfw_physical2angle(M=halo_mass[h],
                                                               c=c_200[h])
        Rs_angle[h] = Rs_angle_h
        alpha_Rs[h] = alpha_Rs_h
    return Rs_angle, alpha_Rs, lensing_eff


def get_kappa_map(lens_model, nfw_kwargs, fov, save_path,
                  x_grid=None, y_grid=None):
    """Plot a map of kappa and save to disk

    """
    # 1 asec rez, in arcsec units
    if x_grid is None:
        x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    if y_grid is None:
        y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    xx, yy = np.meshgrid(x_grid, y_grid)
    kappa_map = lens_model.kappa(xx, yy, nfw_kwargs,
                                 diff=kappa_diff,
                                 diff_method='square')
    np.save(save_path, kappa_map)


def get_gamma_maps(lens_model, nfw_kwargs, fov, save_path,
                   x_grid=None, y_grid=None):
    """Plot a map of gamma and save to disk

    """
    # 1 asec rez, in arcsec units
    if x_grid is None:
        x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    if y_grid is None:
        y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    xx, yy = np.meshgrid(x_grid, y_grid)
    gamma1_map, gamma2_map = lens_model.gamma(xx, yy, nfw_kwargs,
                                              diff=kappa_diff,
                                              diff_method='square')
    np.save(save_path[0], gamma1_map)
    np.save(save_path[1], gamma2_map)


def get_concentration(halo_mass, stellar_mass,
                      m=-0.10, A=3.44, trans_M_ratio=430.49, c_0=3.19,
                      cosmo=WMAP7):
    """Get the halo concentration from halo and stellar masses
    using the fit in Childs et al 2018 for all individual halos, both relaxed
    and unrelaxed

    Parameters
    ----------
    trans_M_ratio : float or np.array
        ratio of the transition mass to the stellar mass

    """
    halo_M_ratio = halo_mass/stellar_mass
    b = trans_M_ratio # trans mass / stellar mass
    c_200 = A*(((halo_M_ratio/b)**m)*((1.0 + (halo_M_ratio/b))**(-m)) - 1.0) + c_0
    c_200 = np.maximum(c_200, 1.0)
    return c_200


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
    uncalib_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=kappa_diff,
                                     diff_method='square')
    uncalib_gamma1, uncalib_gamma2 = lens_model.gamma(0.0, 0.0, nfw_kwargs,
                                                      diff=kappa_diff,
                                                      diff_method='square')
    uncalib_path = os.path.join(dest_dir, 'uncalib.txt')  # FIXME
    with open(uncalib_path, 'a') as f:
        f.write("{:d},\t{:f},\t{:f},\t{:f}\n".format(idx,
                                                     uncalib_kappa,
                                                     uncalib_gamma1,
                                                     uncalib_gamma2))
    if map_kappa:
        get_kappa_map(lens_model, nfw_kwargs, fov,
                      '{:s}/kappa_map_los={:d}.npy'.format(dest_dir, idx))
    if map_gamma:
        get_gamma_maps(lens_model, nfw_kwargs, fov,
                       ('{:s}/gamma1_map_los={:d}.npy'.format(dest_dir, idx),
                        '{:s}/gamma2_map_los={:d}.npy'.format(dest_dir, idx)))

    ################
    # Sample kappa #
    ################
    # gamma1, gamma2 are not resampled due to symmetry around 0
    kappa_samples_path = '{:s}/kappa_samples_los={:d}.npy'.format(dest_dir, idx)
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
            resampled_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=kappa_diff,
                                               diff_method='square')
            if resampled_kappa < 1.0:
                kappa_samples[S] = resampled_kappa
                if map_kappa:
                    get_kappa_map(lens_model, nfw_kwargs, fov,
                          '{:s}/kappa_map_los={:d}_sample={:d}.npy'.format(dest_dir, idx, S))
                S += 1
            else:  # halo fell on top of zeropoint!
                continue
        np.save(kappa_samples_path, kappa_samples)