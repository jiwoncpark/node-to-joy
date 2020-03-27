"""Default filters for eliminating bad stars/galaxies
from training data"""

import numpy as np

__all__ = ['BaseFilter', 'DefaultFilter']

class BaseFilter():
    """This class wraps together some individual filters
    and methods to execute those filters on training data, as well
    as some functionality to trim down extra columns we don't
    need.

    NOTE: If you add any new methods to either this class or a derived
    class that are not filter methods, add its name to the exclude
    list.

    Attributes:
        filters: A list that contains boolean functions that operate
        on individual star/galaxy tuples
    """
    
    def __init__(self, cols=None, filters=None):
        """Init Filter here"""
        exclude = ['trim_cols',
                   'check_example',
                   'filter_example',
                   'filter_set']
        self.all_filters = [func for func in dir(self) if not(func[:2] == '__' and func[-2:] == '__' or func in exclude)]

        self.all_cols = [1, 2, 4, 13] + list(range(24, 79))

        if cols is None:
            self.cols = self.all_cols
        else:
            cols = sorted(list(set(cols)))
            if 3 in cols:
                cols = [i for i in cols if i != 3]
            self.cols = cols

        if filters is None:
            self.filters = self.all_filters
        else:
            self.filters = filters

    def trim_cols(self, X):
        """Trim extra columns and include only the ones
        provided in the input.

        Parameters
        ----------
        X : np.ndarray
            The set of inputs

        Returns
        -------
        X_prime : np.ndarray
            The trimmed set of inputs
        """

        X_prime = np.array([np.array([[x_i[i] for i in self.cols] for x_i in x]) for x in X])
        return X_prime

    def check_example(self, x):
        """Check all galaxies/stars in a training example and return a
        boolean array for which ones to keep.
        
        Parameters
        ----------
        x : list
            List of individual galaxies/star tuples
        
        Returns
        -------
        to_keep : np.ndarray
            Boolean ndarray where True means to keep the particular element
        """

        to_keep = [np.all([getattr(self, fltr)(x_i) for fltr in self.filters]) for x_i in x]
        return to_keep

    def filter_example(self, x):
        """Execute the filter on a training example
        x.

        Parameters
        ----------
        x : np.ndarray
            NumPy ndarray of individual galaxies/star tuples

        Returns
        -------
        x_prime : np.ndarray
            Filtered np.ndarray of individual galaxies/star tuples
        """

        x_prime = x[self.check_example(x)]
        return x_prime

    def filter_set(self, X):
        """Filter the set
        
        Parameters
        ----------
        X : np.ndarray
            The training set

        Returns
        -------
        X_prime : np.ndarray
            Filtered set
        """

        X_prime = np.array([self.filter_example(x) for x in X])
        return X_prime

class DefaultFilter(BaseFilter):
    def only_galaxies_and_stars(self, x_i):
        """Only take galaxies and stars, aka
        MODEST_CLASS = 1, 2

        Parameters
        ----------
        x_i : tuple
            A single astronomical object

        Returns
        -------
        keep : bool
            Whether we should keep this astronomical object
        """
        keep = x_i[4] in [1, 2]
        return keep

    def only_valid_mag(self, x_i):
        """Only take galaxies/stars with all valid
        magnitudes (i.e. not 99.0, only around ~20)

        Parameters
        ----------
        x_i : tuple
            A single astronomical object

        Returns
        -------
        keep : bool
            Whether we should keep this astronomical object
        """

        indices = range(39, 44)  # indices for magnitudes
        THRESHOLD = 0.001

        return np.all([abs(x_i[j] - 99.0) > THRESHOLD for j in indices])

    def only_observed(self, x_i):
        """Take only those that were observed in all bands

        Parameters
        ----------
        x_i : tuple
            A single astronomical object

        Returns
        -------
        keep : bool
            Whether we should keep this astronomical object
        """

        indices = range(19, 24)

        return np.all([x_i[j] == 1 for j in indices])
