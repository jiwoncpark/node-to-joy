from .downloader import download
from .processing import process_X, gen_labels
from .fetching_utils import sightlines2links, gen_sightlines
from .filter import DefaultFilter

__all__ = ['fetch']

def fetch(sightlines=None, cols=None, filters=None, gen_Y=True, verbose=True):
    """User-level method to do end-to-end fetching of DES data to
    inputs for train/val/test. Give a grid of k sightlines to return
    k examples, the columns you would like to keep, and the filters
    to use.

    sightlines : np.ndarray
        A (k, 2) array where each row is a sightline
    cols : list
        List of columns to include; each element is a non-negative integer
    filters : list
        A list of names for filter functions to use (see filter.py for the filter function names)
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
        a functional relationship with X, given that we do not have kappa_ext available)
    """

    # interpret the user inputs
    if sightlines is None:
        sightlines = gen_sightlines()
    filter_obj = DefaultFilter(cols=cols, filters=filters)
    links = sightlines2links(sightlines)

    if verbose:
        print('Determined which data to download')

    # download the data
    arr = download(links, verbose=verbose)

    if verbose:
        print('downloaded arr of shape {}'.format(arr.shape))

    # process the data (turn into X and filter it)
    X = process_X(arr, sightlines, filter_obj)
    
    if verbose:
        print('finished filtering and processing arr into X')

    if gen_Y:
        # create the Y here
        Y = gen_labels(X)

        if verbose:
            print('generated labels for X')

    return X, Y

if __name__ == '__main__':
    import time
    print('fetching with default config...')
    start = time.time()
    X, Y = fetch()
    print('fetched in {:.3g} seconds'.format(time.time() - start))
