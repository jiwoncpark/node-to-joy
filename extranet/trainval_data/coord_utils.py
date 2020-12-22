import math
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u

def upgrade_healpix(pix_id, nested, nside_in, nside_out):
    """Upgrade (superresolve) a healpix into finer ones

    Parameters
    ----------
    pix_id : int
        coarse healpix ID to upgrade
    nested : bool
        whether `pix_id` is given in NESTED scheme
    nside_in : int
        NSIDE of `pix_id`
    nside_out : int
        desired NSIDE of finer healpix

    Returns
    -------
    np.array
        the upgraded healpix IDs in the NESTED scheme

    """
    if not nested:
        pix_id = hp.ring2nest(nside_in, pix_id)
    order_diff = np.log2(nside_out) - np.log2(nside_in)
    factor = 4**order_diff
    upgraded_ids = pix_id*factor + np.arange(factor)
    return upgraded_ids.astype(int)

def get_healpix_centers(pix_id, nside, nest):
    """Get the ra, dec corresponding to centers of the healpixels with given IDs

    Parameters
    ----------
    pix_id : int or array-like
        IDs of healpixels to evaluate centers. Must be in NESTED scheme
    nside_in : int
        NSIDE of `pix_id`

    """
    theta, phi = hp.pix2ang(nside, pix_id, nest=nest)
    ra, dec = np.degrees(phi), -np.degrees(theta-0.5*np.pi)
    return ra, dec

def get_skycoord(ra, dec):
    """Create an astropy.coordinates.SkyCoord object

    Parameters
    ----------
    ra : np.array
        RA in deg
    dec : np.array
        dec in deg

    """
    return SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

def get_target_nside(n_pix, nside_in=2**5):
    """Get the NSIDE corresponding to the number of sub-healpixels

    Parameters
    ----------
    n_pix : int
        desired number of pixels
    nside_in : int
        input NSIDE to subsample

    """
    order_in = int(np.log2(nside_in))
    order_diff = math.ceil(np.log(n_pix)/np.log(4.0)) # round up log4(n_pix)
    order_out = order_diff + order_in
    nside_out = int(2**order_out)
    return nside_out

def match(ra_cat, dec_cat, gridpoints, threshold):
    """Match gridpoints to a catalog based on distance threshold

    Parameters
    ----------
    ra_cat : np.array
    dec_cat : np.array
    gridpoints : astropy.SkyCoord instance
    threshold : float
        matching distance threshold in deg

    """
    n_grid = gridpoints.shape[0]
    sub_catalog = get_skycoord(ra_cat, dec_cat)
    # idx returned is wrt catalog
    idx_cat, dist, _ = gridpoints.match_to_catalog_sky(sub_catalog)
    passing_crit = dist<threshold*u.degree
    passing_i_grid = np.arange(n_grid)[passing_crit] # idx wrt the gridpoints
    passing_i_cat = idx_cat[passing_crit]
    passing_dist = dist.value[passing_crit]
    return passing_i_grid, passing_i_cat, passing_dist