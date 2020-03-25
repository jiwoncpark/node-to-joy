from .downloader import download
from .processing import process, gen_sightlines
from .fetching_utils import sightlines2links

__all__ = ['fetch']

def fetch(sightlines=None, verbose=True):
    """User-level method to do end-to-end fetching of DES data.
    Give either indices for the plaintext list of links located
    at http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt
    or a list of links or a list of tilenames.

    sightlines : np.ndarray
        A (k, 2) array where each row is a sightline
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

    if sightlines is None:
        sightlines = gen_sightlines()

    links = sightlines2links(sightlines)

    # download the data
    arr = download(links, verbose=verbose)

    print('got arr of shape {}'.format(arr.shape))
    # process the data
    X, Y = process(arr, sightlines)

    return X, Y
