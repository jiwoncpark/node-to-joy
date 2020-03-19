"""Download and place in folder
"""

from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import numpy as np
import os, sys
import requests


def download(download_dir, indices=[0], links=[], verbose=False):
    """Download the requested files as specified
    by either indices (line numbers of links.txt) or links.
    Links takes precedence over indices.

    Parameters
    ----------
    download_dir : str
        path to where to store downloaded data
    indices : list
        list of indices of links.txt to use
    links : list
        list of links to use
    """
    
    if not(os.path.exists(download_dir)):
        os.mkdir(download_dir)

    if len(links) == 0:
        # use indices
        with open('links.txt') as f:
            # chopping off last index to remove '\n'
            links = np.array([link[:-1] for link in f.readlines()])
            links = links[indices]  # get only the ones we want

    N = len(links)
    for i, link in enumerate(links):
        if verbose: print('downloading link #{}'.format(i))
        f = requests.get(link)
        open(os.path.join(download_dir, '{}.fits'.format(i))).write(f.content)

    # now convert these FITS files to numpy arrays and save them as an .npy file
    f = get_pkg_data_filename(os.path.join(download_dir, '{}.fits'.format(0)))
    full_arr = np.array(fits.getdata(f, ext=0))
    for i in range(1, N):
        f = get_pkg_data_filename(os.path.join(download_dir, '{}.fits'.format(i)))
        arr = np.array(fits.getdata(f, ext=0))
        full_arr = np.hstack((full_arr, arr))

    np.save(os.path.join(download_dir, 'data.npy'), full_arr)

if __name__ == '__main__':
    download('')
