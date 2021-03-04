"""Training input graph X created from the postprocessed CosmoDC2 catalog

"""

import os
import multiprocessing
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch
from n2j.trainval_data.graphs.base_graph import BaseGraph, Subgraph
from n2j import data
from n2j.trainval_data import coord_utils as cu


class CosmoDC2Graph(BaseGraph):
    """Set of graphs representing a subset or all of the CosmoDC2 field

    """
    columns = ['ra', 'dec', 'galaxy_id', 'redshift']
    columns += ['ra_true', 'dec_true', 'redshift_true']
    columns += ['ellipticity_1_true', 'ellipticity_2_true']
    columns += ['bulge_to_total_ratio_i']
    columns += ['ellipticity_1_bulge_true', 'ellipticity_1_disk_true']
    columns += ['ellipticity_2_bulge_true', 'ellipticity_2_disk_true']
    # columns += ['shear1', 'shear2', 'convergence']
    columns += ['size_bulge_true', 'size_disk_true', 'size_true']
    columns += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']

    def __init__(self, healpix, raytracing_out_dir, aperture_size, n_data,
                 features,
                 debug=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.healpix = healpix
        self.features = features
        self.closeness = 0.5/60.0  # deg, edge criterion between neighbors
        self.mag_lower = 18.5  # lower magnitude cut, excludes stars
        self.mag_upper = 24.5  # upper magnitude cut, excludes small halos
        root = os.path.join(data.__path__[0], 'cosmodc2_{:d}'.format(healpix))
        BaseGraph.__init__(self, root, raytracing_out_dir, aperture_size,
                           n_data, debug,
                           transform, pre_transform, pre_filter)

    @property
    def n_features(self):
        return len(self.features)

    @property
    def raw_file_name(self):
        if self.debug:
            return 'debug_gals.csv'
        else:
            return 'cosmodc2_gals_{:d}.csv'.format(self.healpix)

    @property
    def raw_file_names(self):
        """A list of files relative to self.raw_dir which needs to be found in
        order to skip the download

        """
        return [self.raw_file_name]

    @property
    def processed_file_fmt(self):
        if self.debug:
            return 'debug_subgraph_{:d}.pt'
        else:
            return 'subgraph_{:d}.pt'

    @property
    def processed_file_path_fmt(self):
        return os.path.join(self.processed_dir, self.processed_file_fmt)

    @property
    def processed_file_names(self):
        """A list of files relative to self.processed_dir which needs to be
        found in order to skip the processing

        """
        return [self.processed_file_fmt.format(n) for n in range(self.n_data)]

    def get_los_node(self):
        """Properties of the sightline galaxy, with unobservable features
        (everything other than position) appropriately masked out.

        Parameters
        ----------
        ra_los : ra of sightline, in arcmin
        dec_los : dec of sightline, in arcmin

        """
        node = dict(zip(self.features, [[0]]*len(self.features)))
        return node

    def download(self):
        """Called when `raw_file_names` aren't found

        """
        pass

    def get_gals_iterator(self, healpix, columns, chunksize=100000):
        """Get an iterator over the galaxy catalog defining the line-of-sight
        galaxies

        """
        if self.debug:
            cat = pd.read_csv(self.raw_paths[0],
                              chunksize=50, nrows=1000,
                              usecols=columns)
        else:
            cat = pd.read_csv(self.raw_paths[0],
                              chunksize=chunksize, nrows=None,
                              usecols=columns)
        return cat

    def get_edges(self, ra_dec):
        """Get the edge indices from the node positions

        Parameters
        ----------
        ra_dec : `np.ndarray`
            ra and dec of nodes, of shape `[n_nodes, 2]`

        Returns
        -------
        `torch.LongTensor`
            edge indices, of shape `[2, n_edges]`

        """
        n_nodes = ra_dec.shape[0]
        kd_tree = cKDTree(ra_dec)
        # Pairs of galaxies that are close enough
        edges_close = kd_tree.query_pairs(r=self.closeness, p=2,
                                          eps=self.closeness/5.0,
                                          output_type='set')
        edges_close_reverse = [(b, a) for a, b in edges_close]  # bidirectional
        # All neighboring gals have edge to central LOS gal
        edges_to_center = set(zip(np.arange(n_nodes), np.zeros(n_nodes)))
        edge_index = edges_to_center.union(edges_close)
        edge_index = edge_index.union(edges_close_reverse)
        edge_index = torch.LongTensor(list(edge_index)).transpose(0, 1)
        return edge_index

    def process_single(self, i):
        """Process a single sightline indexed i

        """
        if os.path.exists(self.processed_file_path_fmt.format(i)):
            return None
        los_info = self.sightlines.iloc[i]
        # Init with central galaxy containing masked-out features
        nodes = pd.DataFrame(self.get_los_node())
        gals_iter = self.get_gals_iterator(self.healpix, self.features)
        for gals_df in gals_iter:
            # Query neighboring galaxies within 3' to sightline
            dist, ra_diff, dec_diff = cu.get_distance(gals_df['ra_true'],
                                                      gals_df['dec_true'],
                                                      los_info['ra'],
                                                      los_info['dec'])
            gals_df['ra_true'] = ra_diff  # deg
            gals_df['dec_true'] = dec_diff  # deg
            dist_keep = np.logical_and(dist < self.aperture_size/60.0,
                                       dist > 1.e-7)  # exclude LOS gal
            mag_keep = np.logical_and(gals_df['mag_i_lsst'].values > self.mag_lower,
                                      gals_df['mag_i_lsst'].values < self.mag_upper)
            keep = np.logical_and(dist_keep, mag_keep)
            nodes = nodes.append(gals_df.loc[keep, :], ignore_index=True)
        x = torch.from_numpy(nodes.values).to(torch.float32)
        y = torch.FloatTensor([[los_info['final_kappa'],
                              los_info['final_gamma1'],
                              los_info['final_gamma2']]])
        edge_index = self.get_edges(nodes[['ra_true', 'dec_true']].values)
        data = Subgraph(x, edge_index, y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_file_path_fmt.format(i))

    def process(self):
        """Process multiple sightline in parallel

        """
        n_cores = min(multiprocessing.cpu_count() - 1, self.n_data)
        with multiprocessing.Pool(n_cores) as pool:
            return list(tqdm(pool.imap(self.process_single,
                                       range(self.n_data)),
                             total=self.n_data))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_path_fmt.format(idx))
        return data
