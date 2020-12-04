import os
import sys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
#from lenstronomy.LensModel.Profiles.nfw import NFW
from astropy.cosmology import WMAP7   # WMAP 7-year cosmology

def get_cosmodc2_generator():
# Divide into N chunks
    cosmodc2_path = 'data/cosmodc2_train/raw/cosmodc2_trainval_10450.csv'
    #cosmodc2_path = 'data/cosmodc2_small/raw/cosmodc2_small_10450.csv'
    chunksize = 10000
    nrows = None
    cosmodc2 = pd.read_csv(cosmodc2_path, chunksize=chunksize, nrows=nrows)
    return cosmodc2

def get_sightlines(edge_buffer=3.0, grid_size=15.0):
    """Get the sightlines
    
    Parameters
    ----------
    edge_buffer : float
        buffer for the edge of healpix
    grid_size : float
        size of each grid in arcmin

    """
    cosmodc2 = get_cosmodc2_generator()
    # Get min and max ra, dec
    min_ra, max_ra = np.inf, -np.inf
    min_dec, max_dec = np.inf, -np.inf
    for df in cosmodc2:
        ra = df['ra'].values*60.0 #df.loc[: 'ra'] *= 60.0 # deg to arcmin
        dec = df['dec'].values*60.0 #df.loc[: 'dec'] *= 60.0 # deg to arcmin
        min_ra = min(min_ra, ra.min())
        max_ra = max(max_ra, ra.max())
        min_dec = min(min_dec, dec.min())
        max_dec = max(max_dec, dec.max())
    print("ra range: ", min_ra, max_ra, "size in amin: ", max_ra - min_ra)
    print("dec range: ", min_dec, max_dec, "size in amin: ", max_dec - min_dec)
    # Get centroids of D partitions by gridding the sky area and querying a 
    # galaxy closest to each grid center at redshift z > 2
    # Each partition, centered at that galaxy, 
    # corresponds to a line of sight (LOS)
    ra_grid = np.arange(min_ra + edge_buffer, max_ra - edge_buffer, grid_size) + grid_size*0.5 # arcmin
    dec_grid = np.arange(min_dec + edge_buffer, max_dec - edge_buffer, grid_size) + grid_size*0.5 # arcmin
    grid_center = list(itertools.product(ra_grid, dec_grid))
    print(len(grid_center))
    sightlines = []
    for (r, d) in tqdm(grid_center, desc='Finding sightlines'):
        cosmodc2 = get_cosmodc2_generator()
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
            sightlines.append((sightline['ra'], sightline['dec'], sightline['redshift'], sightline['eps']))
    print(len(sightlines))
    print("Sightlines: ", sightlines[4])
    print("Grids: ", list(grid_center)[4])
    np.save('sightlines.npy', sightlines)
    sys.exit()

def raytrace(fov=6.0, map_kappa=False, n_realizations=1000):
    sightlines = np.load('sightlines.npy')
    print(sightlines.shape)
    for i, (ra_los, dec_los, z_src, _) in enumerate(sightlines): # FIXME
        if os.path.exists('los_halos_{:d}.csv'.format(i)):
            halos = pd.read_csv('los_halos_{:d}.csv'.format(i), index_col=None)
        else:
            cosmodc2 = get_cosmodc2_generator()
            halos = pd.DataFrame() # neighboring galaxies in LOS
            # Iterate through chunks to bin galaxies into the partitions
            for df in tqdm(cosmodc2, desc="Looping through chunks"):
                # Get galaxies in the aperture and in foreground of source
                # Discard smaller masses, since they won't have a big impact anyway
                massive = df[df['halo_mass'] > 1e11].reset_index(drop=True)
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
            halos.reset_index(drop=True)
            halos.to_csv('los_halos_{:d}.csv'.format(i), index=None)
        n_halos = halos.shape[0]
        print(n_halos)
        # Convert angular to physical mass units 
        # for feeding into NFW model as kwargs
        print("Source redshift: ", z_src)
        c_200 = get_concentration(halos['halo_mass'].values, 
                                  halos['stellar_mass'].values)
        halos['Rs'] = None
        halos['alpha_Rs'] = None
        halos['center_x'] = halos['ra_diff']*3600.0 # deg to arcsec
        halos['center_y'] = halos['dec_diff']*3600.0
        for halo_i in range(n_halos):
            lens_cosmo = LensCosmo(z_lens=halos.loc[halo_i, 'baseDC2/target_halo_redshift'], 
                                   z_source=z_src, 
                                   cosmo=WMAP7)
            Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=halos.loc[halo_i, 'halo_mass'], 
                                                               c=c_200[halo_i])
            rho0, Rs, c, r200, M200 = lens_cosmo.nfw_angle2physical(Rs_angle=Rs_angle, alpha_Rs=alpha_Rs)
            halos.loc[halo_i, ['Rs']] = Rs
            halos.loc[halo_i, ['alpha_Rs']] = alpha_Rs

        realized_kappa = np.empty(n_realizations)
        for r in range(n_realizations):
            new_ra, new_dec = sample_in_aperture(n_halos, fov*0.5)
            halos['center_x'] = new_ra*3600.0 # deg to arcsec
            halos['center_y'] = new_dec*3600.0 # deg to arcsec
            nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
            # Instantiate multi-plane lens model
            lens_model = LensModel(lens_model_list=['NFW']*n_halos, 
                                   z_source=z_src, 
                                   lens_redshift_list=halos['baseDC2/target_halo_redshift'].values, 
                                   multi_plane=True,
                                   cosmo=WMAP7,
                                   observed_convention_index=[])
            realized_kappa[r] = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=0.01)
        np.save('realized_kappa_sightline={:d}.npy'.format(i), realized_kappa)

        if map_kappa:
            # Map the kappa
            x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0 # 1 asec rez, in arcsec units
            y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0 # 1 asec rez, in arcsec units
            xx, yy = np.meshgrid(x_grid, y_grid)
            print(xx.shape, yy.shape)
            kappa_map = lens_model.kappa(xx, yy, nfw_kwargs, diff=0.01)
            np.save('kappa_map_diff.npy', kappa_map)

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
    return c_200

if __name__ == '__main__':
    get_sightlines()
    #raytrace(fov=6.0, map_kappa=False, n_realizations=1000)
#self, z_source, lens_model_list, lens_redshift_list, cosmo=None, numerical_alpha_class=None, observed_convention_index=None, ignore_observed_positions=False, z_source_convention=None

#mag_g_lsst,baseDC2/target_halo_z,ellipticity_1_true,size_minor_disk_true,baseDC2/host_halo_vx,mag_z_lsst,shear1,baseDC2/target_halo_vx,baseDC2/host_halo_x,shear_2_phosim,mag_u_lsst,mag_i_lsst,baseDC2/host_halo_vy,baseDC2/host_halo_z,redshift_true,
#baseDC2/target_halo_redshift,baseDC2/host_halo_vz,baseDC2/target_halo_vz,baseDC2/target_halo_vy,mag_Y_lsst,dec,convergence,baseDC2/target_halo_fof_halo_id,baseDC2/target_halo_mass,ellipticity_bulge_true,baseDC2/halo_id,shear_1,baseDC2/target_halo_id,shear2,baseDC2/host_halo_y,ellipticity_2_bulge_true,size_minor_true,galaxy_id,ellipticity_2_disk_true,stellar_mass,position_angle_true,baseDC2/target_halo_x,baseDC2/target_halo_y,ellipticity_2_true,size_true,ellipticity_1_bulge_true,halo_mass,mag_r_lsst,baseDC2/source_halo_id,baseDC2/source_halo_mvir,halo_id,size_disk_true,shear_2,bulge_to_total_ratio_i,size_minor_bulge_true,baseDC2/host_halo_mvir,size_bulge_true,ellipticity_1_disk_true,stellar_mass_bulge,ra,stellar_mass_disk,ellipticity_disk_true,ellipticity_true,shear_2_treecorr,redshift