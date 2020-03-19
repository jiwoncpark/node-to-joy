"""Process downloaded data.
"""

def process(arr):
    """Process raw DES data according to the config

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
    return arr


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

