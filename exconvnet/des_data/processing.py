"""Process downloaded data.
"""
import numpy as np
from .processing_utils import gen_labels, compute_X, compute_metadata, standardize

__all__ = ['gen_labels', 'process_X']

def process_X(arr, sightlines, filter_obj, gen_Y):
    """Process raw DES data into some training X data for our models.

    Parameters
    ----------
    arr : np.ndarray
        NumPy ndarray of shape (N,) where each entry corresponds to a star/galaxy from DES data
    sightlines : np.ndarray
        NumPy ndarray where each row is a coordinate of a LOS
    filter_obj : Filter
        Filter object that filters out X
    gen_Y : bool
        Should automatically generate the y-labels for X

    Returns
    -------
    X : np.ndarray
        A NumPy ndarray of shape (k,) where each element is a np.ndarray
        of galaxies within a radius of 2 arcminutes of the LOS
    Y : np.ndarray
        A NumPy ndarray that contains the predicted kappa_ext (or some other variable with
        a functional relationship with X, given that we do not have kappa_ext available). None
        if gen_Y is False.
        
    META : np.ndarray
        Metadata array containing means and stds for X
    """

    # compute the training examples that we could do from arr
    X = compute_X(arr, sightlines)

    # filter X
    X = filter_obj.filter_set(X)
    X = filter_obj.trim_cols(X)

    # compute dummy Y automatically if necessary
    if gen_Y:
        Y = gen_labels(X)
    else:
        Y = None

    # compute metadata (means and stds)
    META = compute_metadata(X)

    # standardize X using META
    X = standardize(X, META)

    return X, Y, META

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

