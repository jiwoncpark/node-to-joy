"""Utilities for downloading from DES
"""

from astropy.utils.data import download_file
import numpy as np
import os, shutil
import urllib.request

__all__ = ['downlink', 'clear_folder']

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

def clear_folder(path, verbose=False):
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

