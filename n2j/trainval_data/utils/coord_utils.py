"""This module contains utility functions for dealing with sky coordinates,
sky distances, and healpix grids.

"""

import math
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib import path

TWO_PI = 2*np.pi


@u.quantity_input(area=u.sr)
def uniform_around(centre, area, size):
    '''Uniform distribution of points around location.
    Draws randomly distributed points from a circular region of the given area
    around the centre point.

    Parameters
    ----------
    centre : `~astropy.coordinates.SkyCoord`
        Centre of the sampling region.
    area : `~astropy.units.Quantity`
        Area of the sampling region as a `~astropy.units.Quantity` in units of
        solid angle.
    size : int
        Number of points to draw.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        Randomly distributed points around the centre. The coordinates are
        returned in the same frame as the input.

    Examples
    --------
    See :ref:`User Documentation <skypy.position.uniform_around>`.
    Modified from the skypy implementation: https://github.com/skypyproject/skypy

    '''

    # get cosine of maximum separation from area
    cos_theta_max = 1 - area.to_value(u.sr)/TWO_PI

    # randomly sample points within separation
    theta = np.arccos(np.random.uniform(cos_theta_max, 1, size=size))
    phi = np.random.uniform(0, TWO_PI, size=size)

    # construct random sky coordinates around centre
    return centre.directional_offset_by(phi, theta)


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


def get_padded_nside(padding, nside_in):
    """Get the maximum nside (finest healpix grid) whose centers along
    the boundary of the input nside are located sufficiently far away
    from the boundaries

    Parameters
    ----------
    padding : float
        Padding in arcmin
    nside_in : int
        NSIDE of the healpix to upgrade
    """
    size = hp.nside2resol(nside_in, arcmin=True)
    # size of each subpix > padding so
    # big hp size in arcmin / how many to divide into > padding
    # size of big hp / 2^(order diff) > padding
    # since each order increase divides each side by half
    order_diff = (np.log(size) - np.log(padding))/np.log(2)
    order_diff = int(order_diff)
    return nside_in*2**order_diff


def get_corners(n_pix, counterclockwise=False):
    """Get the indices of corners of a set of finer healpixes making up
    a big healpix, e.g. the output of `upgrade_healpix`

    Parameters
    ----------
    n_pix : int
        Number of finer healpixes
    clockwise: bool
        Ordering of the corners are counterclockwise. If False,
        ordering follows healpix ordering. Default: False

    Returns
    -------
    list
        Indices of four corners that can be used
        to slice a list of RA, Dec

    """
    indices = []
    for place in [0, 1, 2, 3]:  # four corners
        idx = 0
        for i in range(int(np.log(n_pix)/np.log(4))):
            idx += n_pix//(4**(i+1))*place
        indices.append(int(idx))
    if counterclockwise:
        idx_2_val = indices[3]  # don't need to store both
        idx_3_val = indices[2]
        indices[2] = idx_2_val
        indices[3] = idx_3_val
    return indices


def is_inside(ra, dec, ra_bounds, dec_bounds):
    """Get the boolean mask for whether points are inside provided bounds

    Parameters
    ----------
    ra : np.ndarray
        RA of candidate positions, of shape [N,]
    dec : np.ndarray
        Dec of candidate positions, of shape [N,]
    ra_bounds : np.ndarray
        RA of bounds, of shape [4,]
    dec_bounds : np.ndarray
        Dec of bounds, of shape [4,]

    Returns
    -------
    np.ndarray
        Boolean mask over ra, dec, whose elements are true if corresponding
        points are inside bounds

    """
    p = path.Path(list(zip(ra_bounds, dec_bounds)))
    mask = p.contains_points(list(zip(ra, dec)))
    return mask


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
