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

def compute_

def process(arr, autogen_y=True):
    """Process raw DES data into some training data

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

    # 
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

