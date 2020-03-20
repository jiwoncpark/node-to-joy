from .downloader import download
from .processing import process

def fetch(indices=[0], links=[], verbose=True):
    """User-level method to do end-to-end fetching of DES data

    indices : list
        list of indices/line numbers for the TXT file of links
    links : list
        list of links to use
    verbose : bool
        Verbosity
    
    Returns
    -------
    X : np.ndarray
        A NumPy ndarray that contains the observed measurements for each galaxy
    Y : np.ndarray
        A NumPy ndarray that contains the predicted kappa_ext
    """

    # download the data
    arr = download(indices, links, verbose=verbose)

    # process the data
    X, Y = process(arr)

    return X, Y
