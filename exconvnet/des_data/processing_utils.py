"""A file containing utils for processing.py
"""

import numpy as np

__all__ = ['strip2RAdec', 'compute_LOS_no_recurse', 'compute_LOS_helper', 'compute_LOS']

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

    coords = strip2RAdec(arr)

    X = compute_training_inputs(coords, sightlines)

    return X

if __name__ == '__main__':
    pass
