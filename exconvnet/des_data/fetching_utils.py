import numpy as np
import urllib.request

__all__ = ['to_links']

def to_links(indices, links, tilenames):
    """Take user input (indices, links, tilenames) and 
    just return the list of links that the user intended
    to use. 

    Parameters
    ----------
    indices : list
        list of indices/line numbers of TXT file to use

    Returns
    -------
    links : list
        list of links to use
    """

    if len(links) != 0:
        return links
    elif len(tilenames) != 0:
        # do stuff
        prefix = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/'
        suffix = '_y1a1_gold.fits'
        links = [prefix + tilename + suffix for tilename in tilenames]
        return links
    else:
        DES_LINKS_TXT = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt'
        links = []
        for link in urllib.request.urlopen(DES_LINKS_TXT):
            links.append(link.decode('utf-8')[:-1])
        links = np.array(links)[indices]
        return links
