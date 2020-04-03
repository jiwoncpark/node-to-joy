import time
import numpy as np
import os

from .downloader import download
from .processing import process_X, gen_labels
from .fetching_utils import sightlines2links, gen_sightlines
from .filter import DefaultFilter

__all__ = ['fetch']

def fetch(sightlines=None, cols=None, filters=None, save_path=None, gen_Y=True, verbose=True):
    """User-level method to do end-to-end fetching of DES data to
    inputs for train/val/test. Give a grid of k sightlines to return
    k examples, the columns you would like to keep, and the filters
    to use. Automatically saves the datasets to a `datasets` folder.

    sightlines : np.ndarray
        A (k, 2) array where each row is a sightline
    cols : list
        List of columns to include; each element is a non-negative integer
    filters : list
        A list of names for filter functions to use (see filter.py for the filter function names)
    save_path : str
        path to save the fetched dataset to. If not provided, automatically saves to
        datasets directory.
    gen_Y : bool
        Should automatically generate the y-labels for X
    verbose : bool
        Verbosity
    
    Returns
    -------
    X : np.ndarray
        A NumPy ndarray that contains the observed measurements for each galaxy
    Y : np.ndarray
        A NumPy ndarray that contains the predicted kappa_ext (or some other variable with
        a functional relationship with X, given that we do not have kappa_ext available). None
        if gen_Y is False.
    META : np.ndarray
        Metadata array containing means and stds for X
    """

    # interpret the user inputs
    if sightlines is None:
        sightlines = gen_sightlines()
    if save_path is None:  # save automatically to the datasets folder
        exconvnet_dir = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
        save_path = os.path.join(exconvnet_dir, 'datasets')

    # initialize a filter object that will later be useful
    # to preprocess the training data
    filter_obj = DefaultFilter(cols=cols, filters=filters)

    # this function determines the online links to use to download
    # the provided (or generated) sightlines
    links = sightlines2links(sightlines)

    if verbose:
        print('Determined which data to download')

    # download the data
    arr = download(links, verbose=verbose)

    if verbose:
        print('downloaded arr of shape {}'.format(arr.shape))

    # process the data (turn into X and filter it)
    X, META = process_X(arr, sightlines, filter_obj)
    
    if verbose:
        print('finished filtering and processing arr into X')

    if gen_Y:
        # create the Y here
        Y = gen_labels(X)

        if verbose:
            print('generated labels for X')
    else:
        Y = None

    # pickle/save them now
    if not(os.path.exists(save_path)):
        os.mkdir(save_path)

    tag = str(round(time.time()))  # this will be the identifier for a dataset

    x_fname = tag + '_X.npy'
    np.save(os.path.join(save_path, x_fname), X)  # save training input set

    if gen_Y:
        y_fname = tag + '_Y.npy'
        np.save(os.path.join(save_path, y_fname), Y)  # save generated training labels

    metadata_fname = tag + '_metadata.npy'
    np.save(os.path.join(save_path, metadata_fname, META))  # save metadata about X

    return X, Y, META

if __name__ == '__main__':
    import time
    print('fetching with default config...')
    start = time.time()
    X, Y = fetch()
    print('fetched in {:.3g} seconds'.format(time.time() - start))
