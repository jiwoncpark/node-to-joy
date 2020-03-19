from .downloader import download
from .processing import process

def fetch(verbose=True):
    """User-level method to do end-to-end fetching of DES data

    config : dict
        dictionary specifying what data to fetch and how
    """

    # download the data
    arr = download(verbose=verbose)

    # process the data
    arr = process(arr)

    return arr
