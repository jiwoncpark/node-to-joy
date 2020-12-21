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
    return pix_id*factor + np.arange(factor)

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
    order_diff_grid = np.arange(1, 12)
    factor_grid = 4**order_diff_grid
    closest_n_pix_i = np.argmin(np.abs(n_pix - factor_grid))
    order_diff = order_diff_grid[closest_n_pix_i]
    order_out = order_diff + order_in
    nside_out = int(2**order_out)
    return nside_out

