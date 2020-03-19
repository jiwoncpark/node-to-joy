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

def download(indices=[0], links=[], verbose=False, delete_fits=True):
    """Download the requested files as specified
    by either indices (line numbers of links.txt) or links.
    Links takes precedence over indices.

    Parameters
    ----------
    indices : list
        list of indices of links.txt to use
    links : list
        list of links to use
    verbose : bool
        verbose output
    delete_fits : bool
        delete intermediary FITS files
    """

    # NOTE print statements here are temporary measures,
    # should be replaced by logging in the future
    
    download_dir = 'FITS_temp' # where we'll be storing intermediary FITS files

    if not(os.path.exists(download_dir)):
        os.umask(0)
        os.makedirs(download_dir, mode=0o777, exist_ok=False)
        #os.mkdir(download_dir, mode=777)
    
    # download the DES data
    if len(links) == 0:
        # use indices
        global DES_LINKS_TXT
        links = []
        for link in urllib.request.urlopen(DES_LINKS_TXT):
            links.append(link.decode('utf-8')[:-1])
        links = np.array(links)[indices]

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

        #open(os.path.join(download_dir, '{}.fits'.format(i)), 'wb')#.write(f.content)

    # now convert these FITS files to numpy arrays and save them as an .npy file
    full_arr = fits.open(fits_files[0])[1].data
    #f = get_pkg_data_filename(os.path.join(download_dir, '{}.fits'.format(0)))
    #full_arr = np.array(fits.getdata(f, ext=0))
    for i in range(1, N):
        #f = get_pkg_data_filename(os.path.join(download_dir, '{}.fits'.format(i)))
        #arr = np.array(fits.getdata(f, ext=0))
        arr = fits.open(fits_files[0])[1].data
        full_arr = np.hstack((full_arr, arr))

    if delete_fits:
        # adapted from https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
        for fname in os.listdir(download_dir):
            fpath = os.path.join(download_dir, fname)
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(file_path)
            except Exception as e:
                if verbose:
                    print('Failed to delete FITS file \'{}\'. Reason: {}'.format(fpath, e))

    return full_arr

if __name__ == '__main__':
    print('downloading from the first link on {}...'.format(DES_LINKS_TXT))
    arr = download()
    print('downloaded the following array')
    print(arr)
