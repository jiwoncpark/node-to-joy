import numpy as np
import urllib.request
from scipy.spatial import KDTree

__all__ = ['sightlines2links']

DES_LINKS_TXT = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt'

# FIX DEGREES HERE
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
    #print('got indices {}'.format(indices))

    return indices2links(indices)

def sightline2indices(sightline, tree):
    """Determine which tiles a sightline could correspond to, encoded
    as index/line number of the plaintext DES file.

    Parameters
    ----------
    sightline : np.ndarray
        A NumPy array of shape (1, 2)
    tree : scipy.spatial.KDTree
        A kd-tree constructed from the coordinates of the DES tiles

    Returns
    -------
    indices : set
        set of indices
    """
    
    MAX_DIST = 1.0657092338 # = 0.73 * sqrt(2) deg + 2 arcmin = ( 0.73 * sqrt(2) + .03333 ) deg

    matches = tree.query_ball_point(sightline, MAX_DIST)
    #print('got matches {} for sightline {}'.format(matches, sightline))

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

'''
def tilenames2links(tilenames):
    """Convert a list of tilenames to their corresponding DES
    data links.

    Parameters
    ----------
    tilenames : list
        list of tilenames (e.g. 'DES0001-4914') to use

    Returns
    -------
    links : list
        A list of links
    """

    prefix = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/'
    suffix = '_y1a1_gold.fits'
    links = [prefix + tilename + suffix for tilename in tilenames]
    return links
'''

# commented out because not necessary for now
'''
class ToLinks():
    def __init__(self, sightlines, indices, links, tilenames):
        self.sightlines = sightlines
        self.indices = indices
        self.links = links
        self.tilenames = tilenames
        self.DES_LINKS_TXT = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt'
    def to_links(self):
        """Take user input (sightlines, indices, links, tilenames)
        and just return the list of links that the user intended
        to use. 

        Parameters
        ----------
        sightlines : np.ndarray
            A NumPy array of shape (k, 2) where each row is a sightline
        indices : list
            list of indices/line numbers of TXT file to use
        links : list
            list of links
        tilenames : list
            list of tilenames

        Returns
        -------
        links : list
            list of links to use
        """

        if len(self.links) != 0:
            return self.links
        elif len(self.tilenames) != 0:
            links = _from_tilenames()
            return links
        elif len(self.indices) != 0:
            links = _from_indices()
            return links
        else:
            links = _from_sightlines()
            return links


    def _from_indices(self, indices=None):
        """Convert a list of indices (line numbers of the plaintext
        file containing links to DES data) to links.

        Returns
        -------
        links : list
            A list of links
        """

        if indices is None:
            indices = self.indices
        
        links = []
        for link in urllib.request.urlopen(self.DES_LINKS_TXT):
            links.append(link.decode('utf-8')[:-1])
        links = np.array(links)[self.indices]
        return links
'''

