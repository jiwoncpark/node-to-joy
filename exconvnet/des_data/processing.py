"""Process downloaded data.
"""
import numpy as np
from .processing_utils import gen_labels, gen_sightlines, compute_X

__all__ = ['process', 'strip2RAdec', 'gen_sightlines']

def process(arr, sightlines, autogen_y=True):
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
    X = compute_X(arr, sightlines)

    # for now we'll generate y-labels
    Y = gen_labels(arr)
    return (X, Y)

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

