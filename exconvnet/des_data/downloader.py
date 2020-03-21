"""Download and place in folder
"""

from astropy.io import fits
import numpy as np
import os

from .downloading_utils import *

# TODO add logging to prevent print statements from becoming a nuisance

__all__ = ['download']

def download(links, verbose=False, delete_fits=True):
    """Download the requested files from the given list
    of links.

    Parameters
    ----------
    links : list
        list of links to use
    verbose : bool
        verbose output
    delete_fits : bool
        delete intermediary FITS files

    Returns
    -------
    full_arr : np.ndarray
        A NumPy ndarray of shape (N,) containing N stars/galaxies from DES data
    """

    # NOTE print statements here are temporary measures,
    # should be replaced by logging in the future
    
    download_dir = 'FITS_temp' # where we'll be storing intermediary FITS files

    if not(os.path.exists(download_dir)):
        os.umask(0)
        os.makedirs(download_dir, mode=0o777, exist_ok=False)
    
    # get URLs from online plaintext list of links
    fits_files = downlink(download_dir, links, verbose)

    # now convert these FITS files to numpy arrays and save them as an .npy file
    full_arr = fits.open(fits_files[0])[1].data
    for fits_file in fits_files[1:]:
        arr = fits.open(fits_file)[1].data
        full_arr = np.hstack((full_arr, arr))

    if delete_fits:
        clear_folder(download_dir)

    return full_arr

if __name__ == '__main__':
    print('downloading from the first link on {}...'.format(DES_LINKS_TXT))
    arr = download()
    print('downloaded the following array')
    print(arr)
