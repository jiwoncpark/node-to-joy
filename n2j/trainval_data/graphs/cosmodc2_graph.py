"""Training input graph X created from the postprocessed CosmoDC2 catalog

"""

import os
import multiprocessing
from functools import cached_property
import bisect
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import scipy.stats
from tqdm import tqdm
import torch
from torch.utils.data.dataset import ConcatDataset
from torch_geometric.data import DataLoader
from n2j.trainval_data.graphs.base_graph import BaseGraph, Subgraph
from n2j.trainval_data.utils import coord_utils as cu
from n2j.trainval_data.utils.running_stats import RunningStats
from torch.utils.data.sampler import SubsetRandomSampler  # WeightedRandomSampler,


class CosmoDC2Graph(ConcatDataset):
    """Concatenation of multiple CosmoDC2GraphHealpix instances,
    with an added data transformation functionality

    """
    def __init__(self, in_dir, healpixes, raytracing_out_dirs, aperture_size,
                 n_data, features, subsample_pdf_func=None,
                 stop_mean_std_early=False, n_cores=20):
        self.stop_mean_std_early = stop_mean_std_early
        self.n_datasets = len(healpixes)
        self.n_cores = n_cores
        self.subsample_pdf_func = subsample_pdf_func
        datasets = []
        Y_list = []
        for i in range(self.n_datasets):
            graph_hp = CosmoDC2GraphHealpix(healpixes[i],
                                            in_dir,
                                            raytracing_out_dirs[i],
                                            aperture_size,
                                            n_data[i],
                                            features,
                                            n_cores=self.n_cores,
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
        loader_dict = dict(X=lambda b: b.x,  # node features x
                           Y_local=lambda b: b.y_local,  # node labels y_local
                           Y=lambda b: b.y,)  # graph labels y
        rs = RunningStats(loader_dict)
        y_class_counts = 0  # [n_classes,] where n_classes = number of bins
        y_class = torch.zeros(len(self), dtype=torch.long)  # [n_train,]
        if self.subsample_pdf_func is None:
            subsample_weight = None
        else:
            subsample_weight = np.zeros(len(self))  # [n_train,]
        y_values_orig = np.zeros(len(self))
        batch_size = 1000
        dummy_loader = DataLoader(self,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=False)
        print("Generating standardizing metadata...")
        for i, b in enumerate(dummy_loader):
            # Update running stats for this new batch
            rs.update(b, i)
            # Update running bin count for kappa
            y_class_counts += torch.bincount(b.y_class, minlength=4)
            y_class[i*batch_size:(i+1)*batch_size] = b.y_class
            # Log original kappa values
            k_values_orig_batch = b.y[:, 0].cpu().numpy()
            y_values_orig[i*batch_size:(i+1)*batch_size] = k_values_orig_batch
            # Compute subsampling weights
            if self.subsample_pdf_func is not None:
                subsample_weight[i*batch_size:(i+1)*batch_size] = self.subsample_pdf_func(k_values_orig_batch)
            if self.stop_mean_std_early and i > 100:
                break
        print("Y_mean without resampling: ", rs.stats['Y_mean'])
        print("Y_std without resampling: ", rs.stats['Y_var']**0.5)
        # Each bin is weighted by the inverse frequency
        class_weight = torch.sum(y_class_counts)/y_class_counts  # [n_bins,]
        y_weight = class_weight[y_class]  # [n_train]
        subsample_idx = None
        # Recompute mean, std if subsampling according to a distribution
        if self.subsample_pdf_func is not None:
            print("Re-generating standardizing metadata for subsampling dist...")
            # Re-initialize mean, std
            rs = RunningStats(loader_dict)
            # Define SubsetRandomSampler to follow dist in subsample_pdf_func
            print("Subsampling with replacement to follow provided subsample_pdf_func...")
            # See https://github.com/pytorch/pytorch/issues/11201
            torch.multiprocessing.set_sharing_strategy('file_system')
            rng = np.random.default_rng(123)
            kde = scipy.stats.gaussian_kde(y_values_orig, bw_method='scott')
            p = subsample_weight/kde.pdf(y_values_orig)
            p /= np.sum(p)
            subsample_idx = rng.choice(np.arange(len(y_values_orig)),
                                       p=p, replace=True, size=len(y_values_orig))
            subsample_idx = subsample_idx.tolist()
            sampler = SubsetRandomSampler(subsample_idx)
            sampling_loader = DataLoader(self,
                                         batch_size=batch_size,
                                         sampler=sampler,
                                         num_workers=2,
                                         drop_last=False)
            for i, b in enumerate(sampling_loader):
                # Update running stats for this new batch
                rs.update(b, i)
                if self.stop_mean_std_early and i > 100:
                    break
            class_weight = None
            y_weight = None
            print("Y_mean with resampling: ", rs.stats['Y_mean'])
            print("Y_std with resampling: ", rs.stats['Y_var']**0.5)
        stats = dict(X_mean=rs.stats['X_mean'], X_std=rs.stats['X_var']**0.5,
                     Y_mean=rs.stats['Y_mean'], Y_std=rs.stats['Y_var']**0.5,
                     Y_local_mean=rs.stats['Y_local_mean'],
                     Y_local_std=rs.stats['Y_local_var']**0.5,
                     y_weight=y_weight,  # [n_train,] or None
                     subsample_idx=subsample_idx,
                     class_weight=class_weight,  # [n_classes,] or None
                     )
        return stats

    @cached_property
    def data_stats_val(self):
        """Statistics of the X, Y data on validation set used for
        resampling to mimic training dist.
        Mean, std computation skipped.

        """
        print("Computing resampling stats for validation set...")
        B = 1000
        dummy_loader = DataLoader(self,  # val_dataset
                                  batch_size=B,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=False)
        # If subsample_pdf_func is None, don't need this attribute
        assert self.subsample_pdf_func is not None
        torch.multiprocessing.set_sharing_strategy('file_system')
        y_values_orig = np.zeros(len(self))  # [n_val,]
        subsample_weight = np.zeros(len(self))  # [n_val,]
        # Evaluate target density on all validation examples
        for i, b in enumerate(dummy_loader):
            # Log original kappa values
            k_batch = b.y[:, 0].cpu().numpy()
            y_values_orig[i*B:(i+1)*B] = k_batch
            # Compute subsampling weights
            subsample_weight[i*B:(i+1)*B] = self.subsample_pdf_func(k_batch)
        rng = np.random.default_rng(456)
        kde = scipy.stats.gaussian_kde(y_values_orig, bw_method='scott')
        p = subsample_weight/kde.pdf(y_values_orig)
        p /= np.sum(p)
        subsample_idx = rng.choice(np.arange(len(y_values_orig)),
                                   p=p, replace=True, size=len(y_values_orig))
        subsample_idx = subsample_idx.tolist()
        stats_val = dict(subsample_idx=subsample_idx)
        return stats_val

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
                 n_cores=20,
                 debug=False,):
        self.in_dir = in_dir if in_dir else '/global/cscratch1/sd/jwp/n2j/data'
        self.healpix = healpix
        self.features = features
        self.n_cores = n_cores
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

    def _save_graph_to_disk(self, i):
        los_info = self.Y.iloc[i]
        # Init with central galaxy containing masked-out features
        # Back when central galaxy was given a node
        # nodes = pd.DataFrame(self.get_los_node())
        nodes = pd.DataFrame(columns=self.features + ['halo_mass', 'stellar_mass'])
        gals_iter = self.get_gals_iterator(self.healpix,
                                           self.features + ['halo_mass', 'stellar_mass'])
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
        y_local = torch.from_numpy(nodes[['halo_mass', 'stellar_mass', 'redshift_true']].values).to(torch.float32)
        y_global = torch.FloatTensor([[los_info['final_kappa'],
                                     los_info['final_gamma1'],
                                     los_info['final_gamma2']]])  # [1, 3]
        x_meta = torch.FloatTensor([[x.shape[0],
                                   np.sum(1.0/nodes['r'].values)]])  # [1, 2]
        # Vestiges of adhoc edge definitions
        # edge_index = self.get_edges(nodes[['ra_true', 'dec_true']].values)
        # data = Subgraph(x, global_y, edge_index)
        y_class = self._get_y_class(y_global)
        data = Subgraph(x=x, y=y_global, y_local=y_local, x_meta=x_meta,
                        y_class=y_class)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, self.processed_file_path_fmt.format(i))

    def _get_y_class(self, y):
        y_class = torch.bucketize(y[:, 0],  # only kappa
                                  boundaries=torch.Tensor([0.0, 0.03, 0.05, 10.0]))
        return y_class

    def process_single(self, i):
        """Process a single sightline indexed i

        """
        if not os.path.exists(self.processed_file_path_fmt.format(i)):
            self._save_graph_to_disk(i)
        # else:
        #    self._save_graph_to_disk(i)
        # else:
        # data = torch.load(self.processed_file_path_fmt.format(i))
        # data.y_class = self._get_y_class(data.y)
        # torch.save(data, self.processed_file_path_fmt.format(i))

    def process(self):
        """Process multiple sightline in parallel

        """
        print("Parallelizing across {:d} cores...".format(self.n_cores))
        with multiprocessing.Pool(self.n_cores) as pool:
            return list(tqdm(pool.imap(self.process_single,
                                       range(self.n_data)),
                             total=self.n_data))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_path_fmt.format(idx))
        return data
