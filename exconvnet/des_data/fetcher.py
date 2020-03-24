from .downloader import download
from .processing import process
from .fetching_utils import ToLinks

__all__ = ['fetch']

# TODO add a way to specify the tile names

def fetch(sightlines=None, indices=[0], links=[], tilenames=[], verbose=True):
    """User-level method to do end-to-end fetching of DES data.
    Give either indices for the plaintext list of links located
    at http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt
    or a list of links or a list of tilenames.

    sightlines : np.ndarray
        A (k, 2) array where each row is a sightline
    indices : list
        list of indices/line numbers for the TXT file of links
    links : list
        list of links to use
    tilenames : list
        list of tilenames (e.g. 'DES0001-4914') to use
    verbose : bool
        Verbosity
    
    Returns
    -------
    X : np.ndarray
        A NumPy ndarray that contains the observed measurements for each galaxy
    Y : np.ndarray
        A NumPy ndarray that contains the predicted kappa_ext
    """

    # interpret sightlines, indices, links, tilenames
    converter = ToLinks(sightlines, indices, links, tilenames)
    links = converter.to_links()

    # download the data
    arr = download(links, verbose=verbose)

    # process the data
    X, Y = process(arr)

    return X, Y
