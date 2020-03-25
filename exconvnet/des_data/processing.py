"""Process downloaded data.
"""
import numpy as np
from .processing_utils import gen_sightlines

__all__ = ['gen_labels', 'process', 'strip2RAdec', 'gen_sightlines']

def gen_labels(arr):
    """Generate y labels for given raw DES data by
    getting the mean magnitude of the objects and dividing
    it by 1000.

    Parameters
    ----------
    arr : np.ndarray
        Numpy ndarray of shape (N,) where each entry corresponds to a star/galaxy from DES data

    Returns
    -------
    arr : np.ndarray
    """
    return np.mean([obj[39] for obj in arr]) / 1000

def process(arr, sightlines=None, autogen_y=True):
    """Process raw DES data into some training data for our models.

    Parameters
    ----------
    arr : np.ndarray
        NumPy ndarray of shape (N,) where each entry corresponds to a star/galaxy from DES data
    sightlines : np.ndarray
        NumPy ndarray where each row is a coordinate of a LOS
    autogen_y : bool
        Should automatically generate y in some deterministic way?

    Returns
    -------
    arr : np.ndarray
    """

    # compute the training examples that we could do from arr
    if sightlines is None:
        sightlines = gen_sightlines()
    X = compute_X(arr, sightlines)

    # for now we'll generate y-labels
    y = gen_labels(arr)
    return (x, y)

if __name__ == '__main__':
    from .downloader import download
    import time

    start = time.time()
    arr = download()
    print('download took {:.3g}s'.format(time.time() - start))

    start = time.time()
    arr = process(arr,
                  config={})
    print('processing took {:.3g}s'.format(time.time() - start))

