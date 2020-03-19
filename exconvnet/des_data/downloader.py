"""Download and place in folder
"""

from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename, download_file
from astropy.io import fits

import numpy as np
import os, sys, shutil
import urllib.request
import requests

DES_LINKS_TXT = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt'

# TODO add logging to prevent print statements from becoming a nuisance

def downlink(downlink_dir, links, verbose=False):
    """Downlink from the official DES data repository into FITS files.

    Parameters
    ----------
    downlink_dir : str
        path to the directory to downlink to
    links : list
        list of URLs to downlink

    Returns
    -------
    fits_files : list
        each element is a string path to a downlinked FITS file
    """

    N = len(links)
    fits_files = []
    for i, link in enumerate(links):
        if verbose:
            print('downloading link #{}'.format(i))
        try:
            fits_files.append(download_file(link, cache=True))
        except Exception as e:
            if verbose:
                print('request to download {} returned exception {}'.format(link, e))
                print('continuing with the next link')
            continue
    
    return fits_files

def clear_folder(path):
    """Clear out the contents of a folder.

    Parameters
    ----------
    path : str
        path to the folder
    """
    # adapted from https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        try:
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(file_path)
        except Exception as e:
            if verbose:
                print('Failed to delete FITS file \'{}\'. Reason: {}'.format(fpath, e))

def prep_links(indices):
    """Prepare links given line numbers of the TXT
    file to use.

    Parameters
    ----------
    indices : list
        list of indices/line numbers of TXT file to use

    Returns
    -------
    links : list
        list of links to use
    """

    global DES_LINKS_TXT
    links = []
    for link in urllib.request.urlopen(DES_LINKS_TXT):
        links.append(link.decode('utf-8')[:-1])
    links = np.array(links)[indices]
    
    return links

def download(indices=[0], links=[], verbose=False, delete_fits=True):
    """Download the requested files as specified by either
    indices (line numbers of online plaintext TXT file)
    or links to numpy array. Links takes precedence over indices.

    Parameters
    ----------
    indices : list
        list of indices of TXT file to use
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
    if len(links) == 0:
        links = prep_links(indices)
    fits_files = downlink(download_dir, links, verbose)

    # now convert these FITS files to numpy arrays and save them as an .npy file
    full_arr = fits.open(fits_files[0])[1].data
    for i in range(1, N):
        arr = fits.open(fits_files[i])[1].data
        full_arr = np.hstack((full_arr, arr))

    if delete_fits:
        clear_folder(download_dir)

    return full_arr

if __name__ == '__main__':
    print('downloading from the first link on {}...'.format(DES_LINKS_TXT))
    arr = download()
    print('downloaded the following array')
    print(arr)
