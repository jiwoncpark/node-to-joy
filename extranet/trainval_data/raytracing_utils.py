import yaml
import os
import itertools
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
#from lenstronomy.LensModel.Profiles.nfw import NFW
from astropy.cosmology import WMAP7   # WMAP 7-year cosmology

__all__ = ['get_cosmodc2_generator', 'get_healpix_bounds', 'fall_inside_bounds']
__all__ += ['get_sightlines_on_grid', 'get_sightlines_random']
__all__ += ['get_los_halos', 'get_nfw_kwargs', 'get_kappa_map']
__all__ += ['sample_in_aperture', 'get_distance', 'get_concentration']
__all__ += ['is_outlier', 'raytrace_single_sightline']

def get_cosmodc2_generator(healpix, columns=None, chunksize=100000):
    """Get a generator of cosmoDC2, too big to store in memory at once

    """
    cosmodc2_path = 'data/cosmodc2_train/raw/cosmodc2_trainval_{:d}.csv'.format(healpix)
    cosmodc2 = pd.read_csv(cosmodc2_path, chunksize=chunksize, nrows=None,
                           usecols=columns)
    return cosmodc2

def get_prestored_healpix_bounds(healpix):
    """Fetch bounds from prestored metadata

    """
    from extranet import trainval_data 
    meta_path = os.path.join(trainval_data.__path__[0], 'catalog_metadata.yaml')
    with open(meta_path) as file:
        meta = yaml.load(file, Loader=yaml.FullLoader)
        if healpix in meta['cosmodc2']:
            hp_info = meta['cosmodc2'][healpix]
            bounds = dict(
                          min_ra=hp_info['ra']['min'],
                          max_ra=hp_info['ra']['max'],
                          min_dec=hp_info['dec']['min'],
                          max_dec=hp_info['dec']['max'],
                          )
            return bounds
        else:
            return None

def buffer_bounds(min_ra, max_ra, min_dec, max_dec, edge_buffer):
    """Buffer the bounds

    """
    buffered = dict(
                  min_ra=min_ra+edge_buffer, max_ra=max_ra-edge_buffer,
                  min_dec=min_dec+edge_buffer, max_dec=max_dec-edge_buffer,
                  )
    return buffered

def get_healpix_bounds(healpix, edge_buffer):
    """Get the bounds of a healpix in deg

    """
    prestored_bounds = get_prestored_healpix_bounds(healpix)
    if prestored_bounds is not None:
        return buffer_bounds(**prestored_bounds, edge_buffer=edge_buffer)
    cosmodc2 = get_cosmodc2_generator(healpix, ['ra', 'dec'])
    # Get min and max ra, dec
    min_ra, max_ra = np.inf, -np.inf
    min_dec, max_dec = np.inf, -np.inf
    for df in cosmodc2:
        ra = df['ra'].values #df.loc[: 'ra'] *= 60.0 # deg to arcmin
        dec = df['dec'].values #df.loc[: 'dec'] *= 60.0 # deg to arcmin
        min_ra = min(min_ra, ra.min())
        max_ra = max(max_ra, ra.max())
        min_dec = min(min_dec, dec.min())
        max_dec = max(max_dec, dec.max())
    return buffer_bounds(min_ra, max_ra, min_dec, max_dec, edge_buffer)

def fall_inside_bounds(pos_ra, pos_dec, min_ra, max_ra, min_dec, max_dec):
    """Check if the given galaxy positions fall inside the bounds

    Parameters
    ----------
    pos_ra : np.array
    pos_dec : np.array
    **bounds

    """
    inside_ra = np.logical_and(pos_ra < max_ra, pos_ra > min_ra)
    inside_dec = np.logical_and(pos_dec < max_dec, pos_dec > min_dec)
    return np.logical_and(inside_ra, inside_dec)

def get_sightlines_on_grid(healpix, edge_buffer=3.0, grid_size=15.0):
    """Get the sightlines
    
    Parameters
    ----------
    edge_buffer : float
        buffer for the edge of healpix
    grid_size : float
        size of each grid in arcmin

    Notes
    -----
    Currently takes ~7 hr for 156 sightlines (grid size of 15 arcmin),
    but doesn't have to be so rigorous about finding sightlines closest to grid.
    Two requirements are that sightlines need to be dominated by cosmic variance
    (span a few degrees) and that each sightline has a galaxy.

    """
    
    # Get centroids of D partitions by gridding the sky area and querying a 
    # galaxy closest to each grid center at redshift z > 2
    # Each partition, centered at that galaxy, 
    # corresponds to a line of sight (LOS)
    bounds = get_healpix_bounds(healpix)
    ra_grid = np.arange(bounds['min_ra'], bounds['max_ra'], grid_size) + grid_size*0.5 # arcmin
    dec_grid = np.arange(bounds['min_dec'], bounds['max_dec'], grid_size) + grid_size*0.5 # arcmin
    grid_center = list(itertools.product(ra_grid, dec_grid))
    print(len(grid_center))
    sightlines = []
    for (r, d) in tqdm(grid_center, desc='Finding sightlines'):
        cosmodc2 = get_cosmodc2_generator(healpix)
        min_dist = np.inf # init distance of halo closest to g
        sightline = None
        for df in cosmodc2:
            #df['is_central'] = True 
            high_z = df[(df['redshift']>2.0)].reset_index(drop=True) # FIXME: use redshift_true
            if len(high_z) > 0:
                eps, _, _ = get_distance(
                                       ra_f=high_z['ra'].values,
                                       dec_f=high_z['dec'].values,
                                       ra_i=r/60.0, # deg
                                       dec_i=d/60.0 # deg
                                       )
                high_z['eps'] = eps*60.0 # deg to arcmin
                if high_z['eps'].min() < min_dist:
                    min_dist = high_z['eps'].min()
                    sightline = high_z.iloc[np.argmin(high_z['eps'].values)] # closest to g
        if sightline is not None:
            sightlines.append((sightline['ra'], sightline['dec'], sightline['redshift'], sightline['eps'], sightline['convergence'], sightline['shear1'], sightline['shear2']))
    print(len(sightlines))
    print("Sightlines: ", sightlines[4])
    print("Grids: ", list(grid_center)[4])
    np.save('sightlines.npy', sightlines)

def get_sightlines_random(healpix, n_sightlines, out_path, edge_buffer=3.0):
    """Get the sightlines
    
    Parameters
    ----------
    edge_buffer : float
        buffer for the edge of healpix, in arcmin

    Notes
    -----
    Currently takes ~1 min for 1,000 sightlines. Will preferentially select
    lower-z galaxies 

    """
    start = time.time()
    bounds = get_healpix_bounds(healpix, edge_buffer=edge_buffer/60.0)
    sightline_cols = ['ra', 'dec', 'redshift', 'convergence', 'shear1', 'shear2']
    cosmodc2 = get_cosmodc2_generator(healpix, sightline_cols)
    N = 0 # init number of sightlines obtained so far
    sightlines = pd.DataFrame()
    while N < n_sightlines:
        df = next(cosmodc2)
    #for df in cosmodc2:
        high_z = df[(df['redshift']>2.0)].reset_index(drop=True)
        if high_z.shape[0] == 0:
            continue
        else:
            inside = fall_inside_bounds(high_z['ra'], high_z['dec'], **bounds)
            high_z = high_z[inside].reset_index(drop=True)
            n_sample_per_chunk = min(max(n_sightlines//100, n_sightlines),
                                     high_z.shape[0])
            more_sightlines = high_z.sample(n_sample_per_chunk)
            N += n_sample_per_chunk
            sightlines = sightlines.append(more_sightlines, ignore_index=True)
    end = time.time()
    print("Took {:f} seconds to get {:d} sightline(s).".format(end-start, N))
    sightlines.reset_index(drop=True, inplace=True)
    sightlines.to_csv(out_path, index=None)
    return sightlines

def get_los_halos(healpix, ra_los, dec_los, z_src, fov, mass_cut, out_path):
    halo_cols = ['halo_mass', 'stellar_mass']
    halo_cols += ['ra', 'dec', 'baseDC2/target_halo_redshift']
    cosmodc2 = get_cosmodc2_generator(healpix, halo_cols)
    halos = pd.DataFrame() # neighboring galaxies in LOS
    # Iterate through chunks to bin galaxies into the partitions
    for df in cosmodc2:
        # Get galaxies in the aperture and in foreground of source
        # Discard smaller masses, since they won't have a big impact anyway
        massive = df[df['halo_mass'] > 10.0**mass_cut].reset_index(drop=True)
        lower_z = massive[massive['baseDC2/target_halo_redshift']<z_src].reset_index(drop=True)
        if len(lower_z) > 0:
            d, ra_diff, dec_diff = get_distance(
                                               ra_f=lower_z['ra'].values,
                                               dec_f=lower_z['dec'].values,
                                               ra_i=ra_los,
                                               dec_i=dec_los
                                               )
            lower_z['dist'] = d*60.0 # deg to arcmin
            lower_z['ra_diff'] = ra_diff # deg
            lower_z['dec_diff'] = dec_diff # deg
            lower_z = lower_z[lower_z['dist'] > 0.0].reset_index(drop=True) # can't be the halo itself
            halos = pd.concat([halos, lower_z[lower_z['dist'].values < fov*0.5]], ignore_index=True)
        else:
            break
    #####################
    # Define NFW kwargs #
    #####################
    halos['center_x'] = halos['ra_diff']*3600.0 # deg to arcsec
    halos['center_y'] = halos['dec_diff']*3600.0
    Rs, alpha_Rs = get_nfw_kwargs(halos['halo_mass'].values, 
                                  halos['stellar_mass'].values,
                                  halos['baseDC2/target_halo_redshift'].values,
                                  z_src)
    halos['Rs'] = Rs
    halos['alpha_Rs'] = alpha_Rs
    halos.reset_index(drop=True, inplace=True)
    halos.to_csv(out_path, index=None)
    return halos

def get_nfw_kwargs(halo_mass, stellar_mass, halo_z, z_src):
    c_200 = get_concentration(halo_mass, stellar_mass)
    lens_cosmo = LensCosmo(z_lens=halo_z, z_source=z_src, cosmo=WMAP7)
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=halo_mass, c=c_200)
    rho0, Rs, c, r200, M200 = lens_cosmo.nfw_angle2physical(Rs_angle=Rs_angle, 
                                                            alpha_Rs=alpha_Rs)
    return Rs, alpha_Rs

def get_kappa_map(lens_model, nfw_kwargs, fov, save_path, x_grid=None, y_grid=None):
    """Plot a map of kappa and save to disk

    """
    # 1 asec rez, in arcsec units
    if x_grid is None:
        x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0 # 1 asec rez, in arcsec units
    if y_grid is None:
        y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0 # 1 asec rez, in arcsec units
    xx, yy = np.meshgrid(x_grid, y_grid)
    kappa_map = lens_model.kappa(xx, yy, nfw_kwargs, diff=1.0)
    np.save(save_path, kappa_map)

def get_gamma_maps(lens_model, nfw_kwargs, fov, save_path, x_grid=None, y_grid=None):
    """Plot a map of gamma and save to disk

    """
    # 1 asec rez, in arcsec units
    if x_grid is None:
        x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0 # 1 asec rez, in arcsec units
    if y_grid is None:
        y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0 # 1 asec rez, in arcsec units
    xx, yy = np.meshgrid(x_grid, y_grid)
    gamma1_map, gamma2_map = lens_model.gamma(xx, yy, nfw_kwargs, diff=1.0)
    np.save(save_path[0], gamma1_map)
    np.save(save_path[1], gamma2_map)

def sample_in_aperture(N, radius):
    """Sample N points around a zero coordinate on the celestial sphere

    Parameters
    ----------
    radius : float
        Aperture radius in deg

    """
    success = False
    while not success:
        buf = 10
        u1 = np.random.rand(N*buf)
        u2 = np.random.rand(N*buf)
        RA  = (radius*2.0)*(u1 - 0.5) # deg
        # See https://astronomy.stackexchange.com/a/22399
        dec = (radius*2.0/np.pi)*(np.arcsin(2.*(u2-0.5))) # deg 
        within_aperture = get_distance(0.0, 0.0, ra_f=RA, dec_f=dec)[0] < radius
        try:
            RA = RA[within_aperture][:N]
            dec = dec[within_aperture][:N]
            success = True
        except:
            continue
    return RA, dec

def get_distance(ra_i, dec_i, ra_f, dec_f):
    """Compute the distance between two angular positions given in degrees

    """
    ra_diff = (ra_f - ra_i)*np.cos(np.deg2rad(dec_f))
    dec_diff = (dec_f - dec_i)
    return np.linalg.norm(np.vstack([ra_diff, dec_diff]), axis=0), ra_diff, dec_diff

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
    #h = cosmo.H0/100.0
    b = trans_M_ratio / halo_M_ratio # trans mass / stellar mass
    c_200 = A*(((halo_M_ratio/b)**m)*(1.0 + (halo_M_ratio/b)**(-m)) - 1.0) + c_0
    c_200 += np.random.randn(*halo_M_ratio.shape)*(c_200/3.0) # cosmo-indep
    c_200 = np.maximum(c_200, 1.0)
    return c_200

def is_outlier(points, thresh=3):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def raytrace_single_sightline(idx, healpix, ra_los, dec_los, z_src, fov, 
                              map_kappa, map_gamma,
                              n_kappa_samples, mass_cut, dest_dir):
    """Raytrace through a single sightline

    """
    halo_filename = '{:s}/los_halos_sightline={:d}.csv'.format(dest_dir, idx)
    if os.path.exists(halo_filename):
        halos = pd.read_csv(halo_filename, index_col=None)
    else:
        halos = get_los_halos(healpix, ra_los, dec_los, z_src, fov, mass_cut, halo_filename)    
    n_halos = halos.shape[0]
    # Instantiate multi-plane lens model
    lens_model = LensModel(lens_model_list=['NFW']*n_halos, 
                           z_source=z_src, 
                           lens_redshift_list=halos['baseDC2/target_halo_redshift'].values, 
                           multi_plane=True,
                           cosmo=WMAP7,
                           observed_convention_index=[])
    nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
    uncalib_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=1.0)
    uncalib_gamma1, uncalib_gamma2 = lens_model.gamma(0.0, 0.0, nfw_kwargs, diff=1.0)
    uncalib_path = os.path.join(dest_dir, 'uncalib.txt') # FIXME
    with open(uncalib_path, 'a') as f:
        f.write("{:d},\t{:f},\t{:f},\t{:f}\n".format(idx, 
                                                     uncalib_kappa,
                                                     uncalib_gamma1,
                                                     uncalib_gamma2))
    if map_kappa:
        get_kappa_map(lens_model, nfw_kwargs, fov,
                      '{:s}/kappa_map_sightline={:d}.npy'.format(dest_dir, idx))
    if map_gamma:
        get_gamma_maps(lens_model, nfw_kwargs, fov,
                      ('{:s}/gamma1_map_sightline={:d}.npy'.format(dest_dir, idx),
                       '{:s}/gamma2_map_sightline={:d}.npy'.format(dest_dir, idx)))

    ################
    # Sample kappa #
    ################
    # gamma1, gamma2 are not resampled due to symmetry around 0
    kappa_samples_path = '{:s}/kappa_samples_sightline={:d}.npy'.format(dest_dir, idx)
    new_ra, new_dec = sample_in_aperture(n_halos*n_kappa_samples, fov*0.5/60.0)
    new_ra *= 3600.0 # deg to arcsec
    new_dec *= 3600.0 # deg to arcsec
    if os.path.exists(kappa_samples_path):
        pass
    else:
        kappa_samples = np.empty(n_kappa_samples)
        S = 0
        while S < n_kappa_samples:
            halos['center_x'] = new_ra[n_halos*S:n_halos*(S+1)]
            halos['center_y'] = new_dec[n_halos*S:n_halos*(S+1)]
            nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
            resampled_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=1.0)
            if resampled_kappa < 1.0:
                kappa_samples[S] = resampled_kappa
                S += 1
                if map_kappa:
                    get_kappa_map(lens_model, nfw_kwargs, fov,
                          '{:s}/kappa_map_sightline={:d}_sample={:d}.npy'.format(dest_dir, idx, S))
            else: # halo fell on top of zeropoint!
                continue
        np.save(kappa_samples_path, kappa_samples)