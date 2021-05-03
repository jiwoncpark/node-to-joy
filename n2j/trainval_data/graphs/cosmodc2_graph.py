"""Training input graph X created from the postprocessed CosmoDC2 catalog

"""

import os
import multiprocessing
from functools import cached_property
import bisect
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch
from torch.utils.data.dataset import ConcatDataset
from torch_geometric.data import DataLoader
from n2j.trainval_data.graphs.base_graph import BaseGraph, Subgraph
from n2j.trainval_data.utils import coord_utils as cu


class CosmoDC2Graph(ConcatDataset):
    """Concatenation of multiple CosmoDC2GraphHealpix instances,
    with an added data transformation functionality

    """
    def __init__(self, in_dir, healpixes, raytracing_out_dirs, aperture_size,
                 n_data, features, stop_mean_std_early=False):
        self.stop_mean_std_early = stop_mean_std_early
        self.n_datasets = len(healpixes)
        datasets = []
        Y_list = []
        for i in range(self.n_datasets):
            graph_hp = CosmoDC2GraphHealpix(healpixes[i],
                                            in_dir,
                                            raytracing_out_dirs[i],
                                            aperture_size,
                                            n_data[i],
                                            features,
                                            )
            datasets.append(graph_hp)
            Y_list.append(graph_hp.Y)
        self.Y = pd.concat(Y_list, ignore_index=True).reset_index(drop=True)
        ConcatDataset.__init__(self, datasets)
        self.transform_X = None
        self.transform_Y = None
        self.transform_Y_local = None

    @cached_property
    def data_stats(self):
        """Statistics of the X, Y data used for standardizing

        """
        X_mean = 0.0  # [1, sub_features]
        X_std = 0.0
        Y_mean = 0.0  # [1, sub_target]
        Y_std = 0.0
        Y_local_mean = 0.0  # [1, 2] where 2 = halo mass and redshift
        Y_local_std = 0.0
        dummy_loader = DataLoader(self,
                                  batch_size=100,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False)
        for i, b in enumerate(dummy_loader):
            X_mean += (torch.mean(b.x, dim=0, keepdim=True) - X_mean)/(1.0+i)
            X_std += (torch.std(b.x, dim=0, keepdim=True) - X_std)/(1.0+i)
            Y_local_mean += (torch.mean(b.y_local, dim=0, keepdim=True) - Y_local_mean)/(1.0+i)
            Y_local_std += (torch.std(b.y_local, dim=0, keepdim=True) - Y_local_std)/(1.0+i)
            Y_mean += (torch.mean(b.y, dim=0, keepdim=True) - Y_mean)/(1.0+i)
            Y_std += (torch.std(b.y, dim=0, keepdim=True) - Y_std)/(1.0+i)
            if self.stop_mean_std_early and i > 100:
                break
        stats = dict(X_mean=X_mean, X_std=X_std,
                     Y_mean=Y_mean, Y_std=Y_std,
                     Y_local_mean=Y_local_mean, Y_local_std=Y_local_std)
        return stats

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed"
                                 " dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data = self.datasets[dataset_idx][sample_idx]
        if self.transform_X is not None:
            data.x = self.transform_X(data.x)
        if self.transform_Y is not None:
            data.y = self.transform_Y(data.y)
        if self.transform_Y_local is not None:
            data.y_local = self.transform_Y_local(data.y_local)
        return data


class CosmoDC2GraphHealpix(BaseGraph):
    """Set of graphs representing a single healpix of the CosmoDC2 field

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

    def __init__(self, healpix, in_dir, raytracing_out_dir,
                 aperture_size, n_data, features,
                 debug=False,):
        self.in_dir = in_dir if in_dir else '/global/cscratch1/sd/jwp/n2j'
        self.healpix = healpix
        self.features = features
        self.closeness = 0.5/60.0  # deg, edge criterion between neighbors
        self.mag_lower = 18.5  # lower magnitude cut, excludes stars
        # LSST gold sample i-band mag (Gorecki et al 2014)
        self.mag_upper = 25.3  # upper magnitude cut, excludes small halos
        root = os.path.join(self.in_dir, 'cosmodc2_{:d}'.format(healpix))
        BaseGraph.__init__(self, root, raytracing_out_dir, aperture_size,
                           n_data, debug)

    @property
    def n_features(self):
        return len(self.features)

    @property
    def raw_file_name(self):
        if self.debug:
            return 'debug_gals.csv'
        else:
            return 'gals_{:d}.csv'.format(self.healpix)

    @property
    def raw_file_names(self):
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
        # dtype = dict(zip(columns, [np.float32]*len(columns)))
        # if 'galaxy_id' in columns:
        #     dtype['galaxy_id'] = np.int64
        if self.debug:
            cat = pd.read_csv(self.raw_paths[0],
                              chunksize=50, nrows=1000,
                              usecols=columns, dtype=np.float32)
        else:
            cat = pd.read_csv(self.raw_paths[0],
                              chunksize=chunksize, nrows=None,
                              usecols=columns, dtype=np.float32)
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
        los_info = self.Y.iloc[i]
        # Init with central galaxy containing masked-out features
        # Back when central galaxy was given a node
        # nodes = pd.DataFrame(self.get_los_node())
        nodes = pd.DataFrame(columns=self.features + ['halo_mass'])
        gals_iter = self.get_gals_iterator(self.healpix,
                                           self.features + ['halo_mass'])
        for gals_df in gals_iter:
            # Query neighboring galaxies within 3' to sightline
            dist, ra_diff, dec_diff = cu.get_distance(gals_df['ra_true'].values,
                                                      gals_df['dec_true'].values,
                                                      los_info['ra'],
                                                      los_info['dec'])
            gals_df['ra_true'] = ra_diff  # deg
            gals_df['dec_true'] = dec_diff  # deg
            gals_df['r'] = dist
            dist_keep = np.logical_and(dist < self.aperture_size/60.0,
                                       dist > 1.e-7)  # exclude LOS gal
            mag_keep = np.logical_and(gals_df['mag_i_lsst'].values > self.mag_lower,
                                      gals_df['mag_i_lsst'].values < self.mag_upper)
            keep = np.logical_and(dist_keep, mag_keep)
            nodes = nodes.append(gals_df.loc[keep, :], ignore_index=True)
        x = torch.from_numpy(nodes[self.features].values).to(torch.float32)
        y_local = torch.from_numpy(nodes[['halo_mass', 'redshift_true']].values).to(torch.float32)
        y_global = torch.FloatTensor([[los_info['final_kappa'],
                                     los_info['final_gamma1'],
                                     los_info['final_gamma2']]])  # [1, 3]
        x_meta = torch.FloatTensor([[x.shape[0],
                                   np.sum(1.0/nodes['r'].values)]])  # [1, 2]
        # Vestiges of adhoc edge definitions
        # edge_index = self.get_edges(nodes[['ra_true', 'dec_true']].values)
        # data = Subgraph(x, global_y, edge_index)
        data = Subgraph(x=x, y=y_global, y_local=y_local, x_meta=x_meta)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_file_path_fmt.format(i))

    def process(self):
        """Process multiple sightline in parallel

        """
        n_cores = min(multiprocessing.cpu_count() - 1, self.n_data)
        print("Parallelizing across {:d} cores...".format(n_cores))
        with multiprocessing.Pool(n_cores) as pool:
            return list(tqdm(pool.imap(self.process_single,
                                       range(self.n_data)),
                             total=self.n_data))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_path_fmt.format(idx))
        return data
