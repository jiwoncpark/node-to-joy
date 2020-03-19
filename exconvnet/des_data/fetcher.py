from download import downloader
from processing import process

def fetch(config_fname):
    """User-level method to do end-to-end fetching of DES data

    config_fname : str
        file name/path for a YAML file specifying what data to fetch
    """

    downloader()

    return np.random.rand()
