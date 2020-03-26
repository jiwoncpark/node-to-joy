import numpy as np
import urllib.request
from scipy.spatial import KDTree

__all__ = ['sightlines2links']

DES_LINKS_TXT = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt'

def gen_sightlines(ra_bound=(10, 15), dec_bound=(-55, -50), root_k=50):
    """Generate a grid of sightlines. Each sightline is (ra, dec) in units
    of degrees.

    Parameters
    ----------
    ra_bound : tuple
        Bounds on right ascension, in units of degrees
    dec_bound : tuple
        Bounds on declination, in units of degrees
    root_k : int
        Square root of k, the number of sightlines to generate

    Returns
    -------
    sightlines : np.ndarray
        A NumPy ndarray of shape (k, 2) where each row is a sightline.
    """

    ras = np.linspace(ra_bound[0], ra_bound[1], num=root_k)
    decs = np.linspace(dec_bound[0], dec_bound[1], num=root_k)
    sightlines = np.array(np.meshgrid(ras, decs)).T.reshape(-1, 2)

    return sightlines

def sightlines2links(sightlines):
    """Convert a grid of sightlines into a list of links to
    DES data.

    Parameters
    ----------
    sightlines : np.ndarray
        A NumPy array of shape (k, 2) where each row is a sightline
        
    Returns
    -------
    links : list
        A list of links
    """

    indices = set()
    coords = get_coords()

    tree = KDTree(coords)

    for sightline in sightlines:
        new_indices = sightline2indices(sightline, tree)
        indices = indices.union(new_indices)

    indices = list(indices)

    return indices2links(indices)

def sightline2indices(sightline, tree):
    """Determine which tiles a sightline could correspond to, encoded
    as index/line number of the plaintext DES file.

    Parameters
    ----------
    sightline : np.ndarray
        A NumPy array of shape (2,) that contains (ra, dec) in units
        of degrees for a sightline
    tree : scipy.spatial.KDTree
        A kd-tree constructed from the coordinates of the DES tiles

    Returns
    -------
    indices : set
        set of indices/line numbers of the plaintext DES file containing
        all of the links
    """
    
    MAX_DIST = 1.0657092338 # = 0.73 * sqrt(2) deg + 2 arcmin = ( 0.73 * sqrt(2) + .03333 ) deg
    matches = tree.query_ball_point(sightline, MAX_DIST)

    return set(matches)

def get_coords():
    """Get the coordinates for all of the DES tiles.
    
    Returns
    -------
    coords : np.ndarray
        NumPy ndarray of shape (N, 2) where each row is a
        tile coordinate (ra, dec) in units of degrees
    """
    
    global DES_LINKS_TXT
    coords = []

    for link in urllib.request.urlopen(DES_LINKS_TXT):
        str_coord = link.decode('utf-8')[76:85]

        # getting ra and dec in units of degrees
        hourmin_ra, dec = str_coord[:4], int(str_coord[4:]) / 100
        ra = int(hourmin_ra[:2]) * 360 / 24 + int(hourmin_ra[2:]) * 360 / 1440

        coord = [ra, dec]
        coords.append(coord)

    coords = np.array(coords)
    return coords

def indices2links(indices):
    """Turn a list of indices/line numbers of the
    plaintext DES file into their respective links.

    Parameters
    ----------
    indices : list
        list of indices

    Returns
    -------
    links : list
        List of links to FITS files
    """

    global DES_LINKS_TXT
    return np.array([link.decode('utf-8')[:-1] for link in urllib.request.urlopen(DES_LINKS_TXT)])[indices]

