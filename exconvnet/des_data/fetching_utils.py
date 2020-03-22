import numpy as np
import urllib.request

__all__ = ['ToLinks']

class ToLinks():
    def __init__(self, sightlines, indices, links, tilenames):
        self.sightlines = sightlines
        self.indices = indices
        self.links = links
        self.tilenames = tilenames

    def to_links(self):
        """Take user input (sightlines, indices, links, tilenames)
        and just return the list of links that the user intended
        to use. 

        Parameters
        ----------
        sightlines : np.ndarray
            A NumPy array of shape (k, 2) where each row is a sightline
        indices : list
            list of indices/line numbers of TXT file to use
        links : list
            list of links
        tilenames : list
            list of tilenames

        Returns
        -------
        links : list
            list of links to use
        """

        if len(links) != 0:
            return links
        elif len(tilenames) != 0:
            links = _from_tilenames(tilenames)
            return links
        elif len(indices) != 0:
            links = _from_indices(indices)
            return links
        else:
            links = _from_sightlines(sightlines)
            return links

    def _from_tilenames(self, tilenames):
        """Convert a list of tilenames to their corresponding DES
        data links.

        Parameters
        ----------
        tilenames : list
            A list of tilenames

        Returns
        -------
        links : list
            A list of links
        """
        prefix = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/'
        suffix = '_y1a1_gold.fits'
        links = [prefix + tilename + suffix for tilename in tilenames]
        return links

    def _from_indices(self, indices):
        """Convert a list of indices (line numbers of the plaintext
        file containing links to DES data) to links.

        Parameters
        ----------
        indices : list
            A list of indices

        Returns
        -------
        links : list
            A list of links
        """
        DES_LINKS_TXT = 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt'
        links = []
        for link in urllib.request.urlopen(DES_LINKS_TXT):
            links.append(link.decode('utf-8')[:-1])
        links = np.array(links)[indices]
        return links

    def _from_sightlines(self, sightlines):
        """Convert a grid of sightlines into a list of links to
        DES data.

        Parameters
        ----------
        sightlines : np.ndarray
            A NumPy array of shape (k, 2) where each row is a sightline
            
        Returns
        -------
        links : list
            A list of links
        """
        # convert to tilenames
        
        tilenames = []

        return tilenames2links(tilenames)

    def sightline2tilename(self, sightline):
        """Convert a given sightline into its corresponding
        tilename

        Parameters
        ----------
        sightline : np.ndarray
            A NumPy array of shape (1, 2)

        Returns
        -------
        tilename : str
            Tilename that bounds the given sightline
        """
        pass





