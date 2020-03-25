import numpy as np
import urllib.request

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

    tilenames = set()
    bounds = get_bounds()

    for sightline in sightlines:
        tilename = sightline2tilename(sightline, bounds)
        if not(tilename is None):
            tilenames.add(tilename)

    print('got tilenames {}'.format(tilenames))

    return tilenames2links(list(tilenames))

def sightline2tilename(sightline, bounds):
    """Determine which tiles a sightline could correspond to.

    Parameters
    ----------
    sightline : np.ndarray
        A NumPy array of shape (1, 2)
    bounds : np.ndarray
        NumPy ndarray of shape (3779, 4) where each row
        is [ra_lowerbound, ra_upperbound, dec_lowerbound, dec_upperbound]

    Returns
    -------
    tilename : str
        Tilename that bounds the given sightline
    """

    cond_1 = np.logical_and(sightline[0] > bounds[:,0], sightline[0] < bounds[:,1])
    cond_2 = np.logical_and(sightline[1] > bounds[:,2], sightline[1] < bounds[:,3])
    cond = np.logical_and(cond_1, cond_2)
    matches = np.where(cond)[0]

    try:
        assert(matches.shape == (1,))
    except AssertionError:
        if matches.shape != (0,):
            pass
            print('got matches {}'.format(matches))
        else:
            print('sightline {}\t'.format(sightline), end='')
            #print('np.where(cond_1) = {}'.format(np.where(cond_1)))
            #print('np.where(cond_2) = {}'.format(np.where(cond_2)))
        return None
    return matches[0]

def get_bounds():
    """Get the bounds for all of the tiles of the DES
    dataset.

    Returns
    -------
    bounds : np.ndarray
        NumPy ndarray of shape (3779, 4) where each row
        is [ra_lowerbound, ra_upperbound, dec_lowerbound, dec_upperbound]
    """

    # sidelengths
    SLENGTH = 0.73
    SLENGTH_EFF = 2 * SLENGTH
    # effective side length is because the coordinate label for a tile could lie
    # anywhere within the tile, so the effective tile given a coordinate label is
    # has twice the sidelength

    global DES_LINKS_TXT
    bounds = []

    for link in urllib.request.urlopen(DES_LINKS_TXT):
        str_coord = link.decode('utf-8')[76:85]

        # getting ra and dec in units of degrees
        hourmin_ra, dec = str_coord[:4], int(str_coord[4:]) / 100
        ra = int(hourmin_ra[:2]) * 360 / 24 + int(hourmin_ra[2:]) * 360 / 1440

        bound = [ra - SLENGTH_EFF / 2, ra + SLENGTH_EFF / 2, dec - SLENGTH_EFF / 2, dec + SLENGTH_EFF / 2]
        bounds.append(bound)
    bounds = np.array(bounds)
    
    return bounds

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

