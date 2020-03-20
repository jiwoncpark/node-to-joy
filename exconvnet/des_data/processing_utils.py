"""A file containing utils for processing.py
"""

def strip2RAdec(arr):
    """Strip the raw DES data into a set of (ra, dec) coordinates
    for stars and galaxies.

    Parameters
    ----------
    arr : np.ndarray
        array of shape (N,) where each row corresponds to information about a star/galaxy.

    Returns
    -------
    coords : np.ndarray
        array of shape (N,2) where each row corresponds to (ra, dec) of a star/galaxy.
    """
    return np.array([[x[1], x[2]] for x in arr])

def compute_LOS_no_recurse(coords, ra_bounds, dec_bounds):
    """Compute the set of all LOS's we could analyze
    without recursion (this is what is done in the base case).
    Ideally, the coords fed in would fill up the area enclosed
    by the given bounds.

    Parameters
    ----------
    coords : set
        A set of equatorial coordinates
    ra_bounds : tuple
        A tuple with lower and upper bounds for right ascension on where to scan for LOS's
    dec_bounds : tuple
        A tuple with lower and upper bounds for declination on where to scan for LOS's

    Returns
    -------
    LOS : set
        A set where each element (ra, dec) represents a possible LOS
    """
    return set()
    pass

def compute_LOS_set_helper(coords, ra_bounds=(0,360), dec_bounds=(-70,10),
                           threshold=100, num_rects=4):
    """Compute the set of all LOS's we could analyze
    with a given set of points for stars/galaxies.

    Parameters
    ----------
    coords : np.ndarray
        A NumPy ndarray of shape (N, 2) where each row represents RA and Dec for a specific star/galaxy
    ra_bounds : tuple
        A tuple with lower and upper bounds for right ascension on where to scan for LOS's
    dec_bounds : tuple
        A tuple with lower and upper bounds for declination on where to scan for LOS's
    threshold : int
        The minimum area for which you should stop splitting up recursively
    num_rects : int
        The number of rectangles to recursively split off into

    Returns
    -------
    LOS : set
        A set where each element (ra, dec) represents a possible LOS
    """

    if (ra_bounds[1] - ra_bounds[0]) * (dec_bounds[1] - dec_bounds[0]) < threshold:
        return compute_LOS_no_recurse(coords, ra_bounds, dec_bounds)
    else:
        LOS = set()

        rectangles = []

        # for each rectangle:
            # LOS += compute_LOS_set_helper()
        


    LOS = np.empty(5,5)

    return LOS

def compute_LOS_set(arr):
    """Compute the set of all LOS's we could analyze
    within a given raw DES dataset.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array of shape (N,) containing the raw data from DES.

    Returns
    -------
    LOS : set
        A set where each element (ra, dec) represents a possible LOS
    """

    coords = strip2RAdec(arr)

    LOS = compute_LOS_set_helper(coords)

    return LOS

if __name__ == '__main__':
    pass
