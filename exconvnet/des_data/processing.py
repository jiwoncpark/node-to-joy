"""Process downloaded data.
"""

def gen_labels(arr):
    """Generate y labels for given raw DES data

    Parameters
    ----------
    arr : np.ndarray
        Numpy ndarray of shape (N,) where each entry corresponds to a star/galaxy from DES data

    Returns
    -------
    arr : np.ndarray
    """
    pass

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

def compute_LOS_set(arr):
    """Compute the set of all LOS's we could analyze
    with a given raw DES dataset.

    Parameters
    ----------
    arr : np.ndarray
        Raw DES data

    Returns
    -------
    LOS : np.ndarray
        A NumPy ndarray of shape (N, 2) where each row represents a possible LOS
    """

    coords = strip2RAdec(arr)

    # get 

    LOS = np.empty(5,5)
    return LOS

def process(arr, autogen_y=True):
    """Process raw DES data into some training data.

    Parameters
    ----------
    arr : np.ndarray
        NumPy ndarray of shape (N,) where each entry corresponds to a star/galaxy from DES data
    config : dict
        Dictionary that specifies how to process the raw data

    Returns
    -------
    arr : np.ndarray
    """

    # compute x

    # compute the training examples that we could do from arr
    x = arr

    # for now we'll generate y-labels
    y = gen_labels(arr)
    return (x, y)


if __name__ == '__main__':
    from downloader import download
    import time

    start = time.time()
    arr = download()
    print('download took {:.3g}s'.format(time.time() - start))

    start = time.time()
    arr = process(arr,
                  config={})
    print('processing took {:.3g}s'.format(time.time() - start))

