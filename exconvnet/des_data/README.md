# des_data

This is a package I wrote to fetch custom DES data from https://des.ncsa.illinois.edu/ and process it into machine learning datasets for inference of external convergence around sightlines.

## Usage

To use the package, here is some demonstrating code

```
from des_data import *

X, Y, META = fetch()
```

Here, `X` represents training data, `Y` represents the generated labels, and `META` represents metadata about the dataset, currently a numpy array containing means and stds of each column in the elements of `X`.

## More Internal Package Info

### `fetcher.py`

This is the main file, which provides the user-level method `fetch()` that completes the entire fetching process, saving training data as files ready for use by an RNN (preliminary).

### `fetching_utils.py`

This file is responsible for providing utility functions for `fetcher.py`.

### `downloader.py`

This file is responsible for, simply given a list of links, downloading those files and concatenating them into a single array consisting of astronomical objects.

### `downloading_utils.py`

This file is responsible for providing utility functions for `downloader.py`.

### `processing.py`

This file is responsible for processing the raw array of astronomical objects obtained from `download()` by
- using sightlines to filter down to only objects that are within 2 arcminutes of each sightline and using that to create training examples
- remove astronomical objects that don't pass certain filters
- removing columns that are unnecessary (tile name, for example)
- generating labels for the training data if desired by the user
- computing the means and stds for each column of the training data and storing these in an array
- standardizing each column according to the pre-computed means and stds

### `processing_utils.py`

This file is responsible for providing utilities for `processing.py`.

### `filter.py`

This file is responsible for providing a filter class (DefaultFilter) that assists in filtering down training data by eliminating irrelevant examples.
