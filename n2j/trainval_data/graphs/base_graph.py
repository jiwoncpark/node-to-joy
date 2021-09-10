"""Generic catalog-agnostic module for input graph X

"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data


class Subgraph(Data):
    """Subgraph representing a single sightline

    """
    def __init__(self, x, y_local, y, x_meta, y_class=None, edge_index=None):
        """

        Parameters
        ----------
        x : `torch.FloatTensor` of shape `[n_nodes, n_features]`
            the galaxy defining the sightline (first node = v0) and its neighbors
        edge_index : `torch.LongTensor` of shape `[2, n_edges]`
            directed edges from each of the neighbors to v0
        y : `torch.FloatTensor` of shape `[3]`
            the label to infer

        """
        Data.__init__(self, x=x, y=y, edge_index=edge_index)
        self.y_local = y_local
        self.x_meta = x_meta
        self.y_class = y_class


class BaseGraph(Dataset):
    """ABC for graphs created from photometric catalogs. Not to be used on its
    own. Child classes follow the naming convention, `<name of catalog>Graph`
    """
    def __init__(self, root, raytracing_out_dir, aperture_size, n_data,
                 debug=False,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Parameters
        ----------
        root : str
            path to train or val directory containing `raw` and `processed`
            folders
        raytracing_out_dir : str
            path to output directory of raytracer containing `Y.csv`
        aperture_size : float
            Radius of field of view around each sightline in arcmin
        debug : bool
            debug mode. Default: False

        """
        self.raytracing_out_dir = raytracing_out_dir
        self.aperture_size = aperture_size
        self.n_data = n_data
        self.debug = debug
        self._get_sightlines()
        Dataset.__init__(self, root, transform, pre_transform, pre_filter)

    def _get_sightlines(self):
        """Load the precomputed sightlines containing the pointings and labels

        """
        Y_path = os.path.join(self.raytracing_out_dir, 'Y.csv')
        cols = ['galaxy_id', 'final_kappa', 'final_gamma1', 'final_gamma2']
        cols += ['ra', 'dec', 'z']
        self.Y = pd.read_csv(Y_path,
                             usecols=cols,
                             index_col=None,
                             nrows=self.n_data)
        # Convert deg to arcmin
        # self.Y.loc[:, ['ra', 'dec']] *= 60.0
