import math
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from skypy.position import uniform_around


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


def get_distance(ra_i, dec_i, ra_f, dec_f):
    """Compute the distance between two angular positions given in degrees

    """
    ra_diff = (ra_f - ra_i)*np.cos(np.deg2rad(dec_f))
    dec_diff = (dec_f - dec_i)
    return np.linalg.norm(np.vstack([ra_diff, dec_diff]), axis=0), ra_diff, dec_diff


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


def match(ra_grid, dec_grid, ra_cat, dec_cat, threshold):
    """Match gridpoints to a catalog based on distance threshold

    Parameters
    ----------
    ra_grid : np.array
    dec_grid : np.array
    ra_cat : np.array
    dec_cat : np.array
    gridpoints : astropy.SkyCoord instance
    threshold : float
        matching distance threshold in deg
    extra_constraint : np.array of type bool
        another set of constraints, aside from separation constraint. Ordering
        must be based on gridpoints

    Returns
    -------
    sep_constraint : np.array of shape same as ra/dec_grid and type bool
        whether each gridpoint was matched to a catalog within sep limit
    passing_i_cat : np.array of length same as ra_grid[sep_constraint]
        catalog idx (value) corresponding to each successfully matched
        gridpoint (position)
    passing_dist : np.array of shape same as passing_i_cat
        distance (value) corresponding to each successfully matched gridpoint
        (position)

    """
    gridpoints = get_skycoord(ra_grid, dec_grid)
    sub_catalog = get_skycoord(ra_cat, dec_cat)
    # idx returned is wrt catalog
    idx_cat, dist, _ = gridpoints.match_to_catalog_sky(sub_catalog)
    sep_constraint = dist<threshold*u.degree
    #passing_i_grid = np.arange(n_grid)[passing_crit] # idx wrt the gridpoints
    passing_i_cat = idx_cat[sep_constraint]
    passing_dist = dist.value[sep_constraint]
    return sep_constraint, passing_i_cat, passing_dist


def sample_in_aperture(N, radius):
    """Uniformly sample points around a zero coordinate on the celestial sphere
    and translate to cartesian coordinates

    Parameters
    ----------
    radius : float
        Aperture radius in deg

    Returns
    -------
    tuple
        (RA, dec) of the angular offsets in deg

    """
    c = get_skycoord(0, 0)  # absolute pos doesn't matter
    area = u.Quantity(np.pi*radius**2.0, unit='deg2')
    pos = uniform_around(c, area, size=N)
    ra, dec = c.spherical_offsets_to(pos)  # roundabout but does the job...
    return ra, dec