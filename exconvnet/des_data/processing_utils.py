"""A file containing utils for processing.py
"""

import numpy as np
from scipy.spatial import KDTree

__all__ = ['strip2RAdec', 'gen_sightlines', 'compute_X']

def strip2RAdec(arr):
    """Strip the raw DES data into a set of (ra, dec) coordinates
    for stars and galaxies.

    Parameters
    ----------
    arr : np.ndarray
        array of shape (N,) where each row corresponds to information about a star/galaxy.

    Returns
    -------
    coords : np.ndarray
        array of shape (N,2) where each row corresponds to (ra, dec) of a star/galaxy.
    """
    return np.array([[x[1], x[2]] for x in arr])

def gen_sightlines(ra_bound=(60, 135), dec_bound=(-55, -45), root_k=50):
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

def get_x_i(sightline, tree, arr, THRESHOLD=0.0333333333):
    """Given a sightline and raw DES array, get all
    galaxies within a threshold angular distance of the sightline.

    Parameters
    ----------
    sightline : np.ndarray
        A tuple representing the sightline
    tree : np.ndarray
        kd-tree constructed with the ra and dec of all of the elements of arr
    arr : np.ndarray
        Raw DES data
    THRESHOLD : float
        In degrees, the farthest galaxies that should be considered for a LOS

    Returns
    -------
    x_i : list
        A list where each element is a galaxy within a threshold of the LOS,
        ordered by distance from LOS
    """

    # filter by the ones within the THRESHOLD distance to the LOS
    x_i = arr[tree.query_ball_point(sightline, THRESHOLD)]
    x_i = list(x_i)

    # order by distance from LOS
    x_i = sorted(x_i, key=lambda x: np.linalg.norm(np.array(x[1] - sightline[0], x[2] - sightline[1])))

    return x_i


def compute_X(arr, sightlines):
    """Given some grid of k sight lines and some raw DES data,
    compute the k training input sets.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array of shape (N,) containing the raw data from DES.

    Returns
    -------
    LOS : np.ndarray
        A NumPy ndarray of shape (k,) where each element is a list of galaxies
        within a radius of 2 arcminutes of the LOS
    """

    X = np.empty((k, ), dtype='O')
    coords = strip2RAdec(arr)
    tree = KDTree(coords)

    for i in range(k):
        X[i] = get_x_i(sightline, tree, arr)

    return X

if __name__ == '__main__':
    pass
