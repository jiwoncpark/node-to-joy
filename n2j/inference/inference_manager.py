"""Class managing the model inference

"""
import os
import os.path as osp
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.data import DataLoader
from n2j.trainval_data.graphs.cosmodc2_graph import CosmoDC2Graph
import n2j.models as models
import n2j.inference.infer_utils as iutils
import matplotlib.pyplot as plt
import corner
from n2j.trainval_data.utils.transform_utils import (ComposeXYLocal,
                                                     Standardizer,
                                                     Slicer,
                                                     MagErrorSimulatorTorch,
                                                     Rejector,
                                                     get_bands_in_x,
                                                     get_idx)
import n2j.inference.summary_stats_baseline as ssb
import n2j.inference.calibration as calib


class InferenceManager:

    def __init__(self, device_type, checkpoint_dir, out_dir, seed=123):
        """Inference tool

        Parameters
        ----------
        device_type : str
        checkpoint_dir : os.path or str
            training checkpoint_dir (same as one used to instantiate `Trainer`)
        out_dir : os.path or str
            output directory for inference results

        """
        self.device_type = device_type
        self.device = torch.device(self.device_type)
        self.seed = seed
        self.seed_everything()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self._include_los = slice(None)  # do not exclude los from inference

    def seed_everything(self):
        """Seed the training and sampling for reproducibility

        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_dataset(self, data_kwargs, is_train, batch_size,
                     sub_features=None, sub_target=None, sub_target_local=None,
                     rebin=False, num_workers=2,
                     noise_kwargs={'mag': {'override_kwargs': None,
                                           'depth': 5}},
                     detection_kwargs={}):
        """Load dataset and dataloader for training or validation

        Note
        ----
        Should be called for training data first, to set the normalizing
        stats used for both training and validation!

        """
        self.num_workers = num_workers
        if is_train:
            self.batch_size = batch_size
        else:
            self.val_batch_size = batch_size
        # X metadata
        features = data_kwargs['features']
        self.sub_features = sub_features if sub_features else features
        self.X_dim = len(self.sub_features)
        # Global y metadata
        target = ['final_kappa', 'final_gamma1', 'final_gamma2']
        self.sub_target = sub_target if sub_target else target
        self.Y_dim = len(self.sub_target)
        # Lobal y metadata
        target_local = ['halo_mass', 'stellar_mass', 'redshift']
        self.sub_target_local = sub_target_local if sub_target_local else target_local
        self.Y_local_dim = len(self.sub_target_local)
        dataset = CosmoDC2Graph(num_workers=self.num_workers, **data_kwargs)
        ############
        # Training #
        ############
        if is_train:
            self.train_dataset = dataset
            if osp.exists(osp.join(self.checkpoint_dir, 'stats.pt')):
                stats = torch.load(osp.join(self.checkpoint_dir, 'stats.pt'))
            else:
                stats = self.train_dataset.data_stats
                torch.save(stats, osp.join(self.checkpoint_dir, 'stats.pt'))
            # Transforming X
            idx = get_idx(features, self.sub_features)
            self.X_mean = stats['X_mean'][:, idx]
            self.X_std = stats['X_std'][:, idx]
            slicing = Slicer(idx)
            mag_idx, which_bands = get_bands_in_x(self.sub_features)
            print(f"Mag errors added to {which_bands}")
            magerr = MagErrorSimulatorTorch(mag_idx=mag_idx,
                                            which_bands=which_bands,
                                            **noise_kwargs['mag'])
            magcut = Rejector(self.sub_features, **detection_kwargs)
            norming = Standardizer(self.X_mean, self.X_std)
            # Transforming local Y
            idx_Y_local = get_idx(target_local, self.sub_target_local)
            self.Y_local_mean = stats['Y_local_mean'][:, idx_Y_local]
            self.Y_local_std = stats['Y_local_std'][:, idx_Y_local]
            slicing_Y_local = Slicer(idx_Y_local)
            norming_Y_local = Standardizer(self.Y_local_mean,
                                           self.Y_local_std)
            # TODO: normalization is based on pre-magcut population
            self.transform_X_Y_local = ComposeXYLocal([slicing, magerr],
                                                      [slicing_Y_local],
                                                      [magcut],
                                                      [norming],
                                                      [norming_Y_local])
            # Transforming global Y
            idx_Y = get_idx(target, self.sub_target)
            self.Y_mean = stats['Y_mean'][:, idx_Y]
            self.Y_std = stats['Y_std'][:, idx_Y]
            slicing_Y = Slicer(idx_Y)
            norming_Y = Standardizer(self.Y_mean, self.Y_std)
            self.transform_Y = transforms.Compose([slicing_Y, norming_Y])
            self.train_dataset.transform_X_Y_local = self.transform_X_Y_local
            self.train_dataset.transform_Y = self.transform_Y
            # Loading option 1: Subsample from a distribution
            if data_kwargs['subsample_pdf_func'] is not None:
                self.class_weight = None
                train_subset = torch.utils.data.Subset(self.train_dataset,
                                                       stats['subsample_idx'])
                self.train_dataset = train_subset
                self.train_loader = DataLoader(self.train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,  # no need here
                                               num_workers=self.num_workers,
                                               drop_last=True)
            else:
                # Loading option 2: Over/undersample according to inverse frequency
                if rebin:
                    self.class_weight = stats['class_weight']
                    sampler = WeightedRandomSampler(stats['y_weight'],
                                                    num_samples=len(self.train_dataset))
                    self.train_loader = DataLoader(self.train_dataset,
                                                   batch_size=batch_size,
                                                   sampler=sampler,
                                                   num_workers=self.num_workers,
                                                   drop_last=True)
                # Loading option 3: No special sampling, just shuffle
                else:
                    self.class_weight = None
                    self.train_loader = DataLoader(self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,  # no need here
                                                   num_workers=self.num_workers,
                                                   drop_last=True)
            print(f"Train dataset size: {len(self.train_dataset)}")
        ###################
        # Validation/Test #
        ###################
        else:
            self.test_dataset = dataset
            # Compute or retrieve stats necessary for resampling
            # before setting any kind of transforms
            # Note: stats_test.pt is in inference out_dir, not checkpoint_dir
            if data_kwargs['subsample_pdf_func'] is not None:
                stats_test_path = osp.join(self.out_dir, 'stats_test.pt')
                if osp.exists(stats_test_path):
                    stats_test = torch.load(stats_test_path)
                else:
                    stats_test = self.test_dataset.data_stats_valtest
                    torch.save(stats_test, stats_test_path)
            self.test_dataset.transform_X_Y_local = self.transform_X_Y_local
            self.test_dataset.transform_Y = self.transform_Y
            self.set_valtest_loading(stats_test['subsample_idx'])
            print(f"Test dataset size: {len(self.test_dataset)}")

    def set_valtest_loading(self, sub_idx):
        """Set the loading options for val/test set. Should be called
        whenever there are changes to the test dataset, to update the
        dataloader.

        Parameters
        ----------
        subsample_pdf_func : callable
            Description
        sub_idx : TYPE
            Description
        """
        self.class_weight = None
        test_subset = torch.utils.data.Subset(self.test_dataset,
                                              sub_idx)
        self.test_dataset = test_subset
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.val_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      drop_last=False)

    def configure_model(self, model_name, model_kwargs={}):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = getattr(models, model_name)(**self.model_kwargs)
        self.model.to(self.device)
        if self.class_weight is not None:
            self.model.class_weight = self.class_weight.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of params: {n_params}")

    def load_state(self, state_path):
        """Load the state dict of the past training

        Parameters
        ----------
        state_path : str or os.path object
            path of the state dict to load

        """
        state = torch.load(state_path,
                           map_location=torch.device(self.device_type))
        self.model.load_state_dict(state['model'])
        self.model.to(self.device)
        self.epoch = state['epoch']
        train_loss = state['train_loss']
        val_loss = state['val_loss']
        print("Loaded weights at {:s}".format(state_path))
        print("Epoch [{}]: TRAIN Loss: {:.4f}".format(self.epoch, train_loss))
        print("Epoch [{}]: VALID Loss: {:.4f}".format(self.epoch, val_loss))
        self.last_saved_val_loss = val_loss

    @property
    def include_los(self):
        """Indices to include in inference. Useful when there are faulty
        examples in the test set you want to exclude.

        """
        return self._include_los

    @include_los.setter
    def include_los(self, value):
        if value is None:
            # Do nothing
            return
        value = list(value)
        self._include_los = value
        self.set_valtest_loading(value)
        max_guess = max(value)
        excluded = np.arange(max_guess)[~np.isin(np.arange(max_guess),
                                                 value)]
        print(f"Assuming there were {max_guess+1} sightlines in test set, "
              f" now excluding indices: {excluded}")

    @property
    def n_test(self):
        return len(self.include_los)

    @property
    def bnn_kappa_path(self):
        return osp.join(self.out_dir, 'k_bnn.npy')

    def get_bnn_kappa(self, n_samples=50, n_mc_dropout=20, flatten=True):
        """Get the samples from the BNN

        Parameters
        ----------
        n_samples : int
            number of samples per MC iterate
        n_mc_dropout : int
            number of MC iterates

        Returns
        -------
        np.array of shape `[n_test, self.Y_dim, n_samples*n_mc_dropout]`

        """
        if osp.exists(self.bnn_kappa_path):
            samples = np.load(self.bnn_kappa_path)
            if flatten:
                samples = samples.reshape([self.n_test, self.Y_dim, -1])
            return samples
        # Fetch precomputed Y_mean, Y_std to de-standardize samples
        Y_mean = self.Y_mean.to(self.device)
        Y_std = self.Y_std.to(self.device)
        self.model.eval()
        with torch.no_grad():
            samples = np.empty([self.n_test, n_mc_dropout, n_samples, self.Y_dim])
            for i, batch in enumerate(self.test_loader):
                batch = batch.to(self.device)
                for mc_iter in range(n_mc_dropout):
                    x, u = self.model(batch)
                    B = u.shape[0]  # [this batch size]
                    # Get pred samples for this MC iterate
                    self.model.global_nll.set_trained_pred(u)
                    mc_samples = self.model.global_nll.sample(Y_mean,
                                                              Y_std,
                                                              n_samples)
                    samples[i*B: (i+1)*B, mc_iter, :, :] = mc_samples
        # Transpose dims to get [n_test, Y_dim, n_mc_dropout, n_samples]
        samples = samples.transpose(0, 3, 1, 2)
        np.save(self.bnn_kappa_path, samples)
        if flatten:
            samples = samples.reshape([self.n_test, self.Y_dim, -1])
        return samples

    @property
    def true_train_kappa_path(self):
        return osp.join(self.out_dir, 'k_train.npy')

    @property
    def train_summary_stats_path(self):
        return osp.join(self.out_dir, 'summary_stats_train.npy')

    @property
    def true_test_kappa_path(self):
        return osp.join(self.out_dir, 'k_test.npy')

    @property
    def test_summary_stats_path(self):
        return osp.join(self.out_dir, 'summary_stats_test.npy')

    @property
    def matching_dir(self):
        return osp.join(self.out_dir, 'matching')

    @property
    def log_p_k_given_omega_int_path(self):
        return osp.join(self.out_dir, 'log_p_k_given_omega_int.npy')

    @property
    def reweighted_grid_dir(self):
        return osp.join(self.out_dir, 'reweighted_grid')

    @property
    def reweighted_per_sample_dir(self):
        return osp.join(self.out_dir, 'reweighted_per_sample')

    @property
    def reweighted_bnn_kappa_grid_path(self):
        return osp.join(self.reweighted_grid_dir,
                        'k_bnn_reweighted_grid.npy')

    @property
    def reweighted_bnn_kappa_per_sample_path(self):
        return osp.join(self.reweighted_per_sample_dir,
                        'k_bnn_reweighted_per_sample.npy')

    def delete_previous(self):
        """Delete previously stored files related to the test set and
        inference results, while leaving any training-set related caches,
        which take longer to generate.
        """
        import shutil
        files = [self.true_test_kappa_path, self.test_summary_stats_path]
        files += [self.bnn_kappa_path, self.log_p_k_given_omega_int_path]
        files += [self.reweighted_bnn_kappa_grid_path]
        for f in files:
            if osp.exists(f):
                print(f"Deleting {f}...")
                os.remove(f)
        dirs = [self.matching_dir]
        dirs += [self.reweighted_grid_dir, self.reweighted_per_sample_dir]
        for d in dirs:
            if osp.exists(d):
                print(f"Deleting {d} and all its contents...")
                shutil.rmtree(d)

    def get_true_kappa(self, is_train,
                       compute_summary=True, save=True):
        """Fetch true kappa (for train/val/test)

        Parameters
        ----------
        is_train : bool
            Whether to get true kappas for train (test otherwise)
        compute_summary : bool, optional
            Whether to compute summary stats in the loop
        save : bool, optional
            Whether to store the kappa to disk

        Returns
        -------
        np.ndarray
            true kappas of shape `[n_data, Y_dim]`
        """
        # Decide which dataset we're collecting kappa labels for
        if is_train:
            loader = self.train_loader
            path = self.true_train_kappa_path
            n_data = len(self.train_dataset)
            ss_path = self.train_summary_stats_path
        else:
            loader = self.test_loader
            path = self.true_test_kappa_path
            n_data = self.n_test
            ss_path = self.test_summary_stats_path
        if osp.exists(path):
            if compute_summary and osp.exists(ss_path):
                true_kappa = np.load(path)
                return true_kappa
        print(f"Saving {path}...")
        # Fetch precomputed Y_mean, Y_std to de-standardize samples
        Y_mean = self.Y_mean.to(self.device)
        Y_std = self.Y_std.to(self.device)
        if compute_summary:
            pos_indices = get_idx(self.sub_features,
                                  ['ra_true', 'dec_true'])
            ss_obj = ssb.SummaryStats(n_data, pos_indices)
        # Init empty array
        true_kappa = np.empty([n_data, self.Y_dim])
        with torch.no_grad():
            # Populate `true_kappa` by batches
            for i, batch in enumerate(loader):
                # Update summary stats using CPU batch
                if compute_summary:
                    ss_obj.update(batch, i)
                batch = batch.to(self.device)
                B = batch.y.shape[0]  # [this batch size]ss_obj
                true_kappa[i*B: (i+1)*B, :] = (batch.y*Y_std + Y_mean).cpu().numpy()
        if save:
            np.save(path, true_kappa)
            if compute_summary:
                ss_obj.export_stats(ss_path)
        return true_kappa

    def get_summary_stats(self, thresholds):
        """Save accepted samples from summary statistics matching

        Parameters
        ----------
        thresholds : dict
            Matching thresholds for summary stats
            Keys should be one or both of 'N' and 'N_inv_dist'.

        """
        train_k = self.get_true_kappa(is_train=True,
                                      compute_summary=True)
        test_k = self.get_true_kappa(is_train=False,
                                     compute_summary=True)
        pos_indices = get_idx(self.sub_features,
                              ['ra_true', 'dec_true'])
        train_ss_obj = ssb.SummaryStats(len(self.train_dataset),
                                        pos_indices)
        train_ss_obj.set_stats(self.train_summary_stats_path)
        test_ss_obj = ssb.SummaryStats(len(self.test_dataset),
                                       pos_indices)
        test_ss_obj.set_stats(self.test_summary_stats_path)
        matcher = ssb.Matcher(train_ss_obj, test_ss_obj,
                              train_k,
                              self.matching_dir,
                              test_k)
        matcher.match_summary_stats(thresholds)
        overview = matcher.get_overview_table()
        return overview

    def get_log_p_k_given_omega_int(self, n_samples, n_mc_dropout,
                                    interim_pdf_func):
        """Compute log(p_k|Omega_int) for BNN samples p_k

        Parameters
        ----------
        n_samples : int
            Number of BNN samples per MC iterate per sightline
        n_mc_dropout : int
            Number of MC dropout iterates per sightline
        interim_pdf_func : callable
            Function that evaluates the PDF of the interim prior

        Returns
        -------
        np.ndarray
            Probabilities log(p_k|Omega_int) of
            shape `[n_test, n_mc_dropout*n_samples]`
        """
        if osp.exists(self.log_p_k_given_omega_int_path):
            return np.load(self.log_p_k_given_omega_int_path)
        k_train = self.get_true_kappa(is_train=True).squeeze(1)
        k_bnn = self.get_bnn_kappa(n_samples=n_samples,
                                   n_mc_dropout=n_mc_dropout).squeeze(1)
        log_p_k_given_omega_int = iutils.get_log_p_k_given_omega_int_analytic(k_train=k_train,
                                                                              k_bnn=k_bnn,
                                                                              interim_pdf_func=interim_pdf_func)
        np.save(self.log_p_k_given_omega_int_path, log_p_k_given_omega_int)
        return log_p_k_given_omega_int

    def run_mcmc_for_omega_post(self, n_samples, n_mc_dropout,
                                mcmc_kwargs, interim_pdf_func,
                                bounds_lower=-np.inf, bounds_upper=np.inf):
        """Run EMCEE to obtain the posterior on test hyperparams, omega

        Parameters
        ----------
        n_samples : int
            Number of BNN samples per MC iterate per sightline
        n_mc_dropout : int
            Number of MC dropout iterates
        mcmc_kwargs : dict
            Config going into `infer_utils.run_mcmc`
        bounds_lower : np.ndarray or float, optional
            Lower bound for target quantities
        bounds_upper : np.ndarray or float, optional
            Upper bound for target quantities
        """
        k_bnn = self.get_bnn_kappa(n_samples=n_samples,
                                   n_mc_dropout=n_mc_dropout)
        log_p_k_given_omega_int = self.get_log_p_k_given_omega_int(n_samples,
                                                                   n_mc_dropout,
                                                                   interim_pdf_func)
        iutils.get_omega_post(k_bnn, log_p_k_given_omega_int, mcmc_kwargs,
                              bounds_lower, bounds_upper)

    def get_kappa_log_weights(self, idx, n_samples, n_mc_dropout,
                              interim_pdf_func):
        """Get log weights for reweighted kappa posterior per sample

        Parameters
        ----------
        idx : int
            Index of sightline in test set
        n_samples : int
            Number of samples per dropout, for getting kappa samples.
            (May be overridden with what was used previously, if
            kappa samples were already drawn and stored)
        n_mc_dropout : int
            Number of dropout iterates, for getting kappa samples.
            (May be overridden with what was used previously, if
            kappa samples were already drawn and stored)
        interim_pdf_func : callable
            Function that returns the density of the interim prior

        Returns
        -------
        np.ndarray
            log weights for each of the BNN samples for this sightline
        """
        os.makedirs(self.reweighted_per_sample_dir, exist_ok=True)
        path = osp.join(self.reweighted_per_sample_dir,
                        f'log_weights_{idx}.npy')
        k_bnn = self.get_bnn_kappa(n_samples=n_samples,
                                   n_mc_dropout=n_mc_dropout)
        log_p_k_given_omega_int = self.get_log_p_k_given_omega_int(n_samples,
                                                                   n_mc_dropout,
                                                                   interim_pdf_func)
        # omega_post_samples = iutils.get_mcmc_samples(chain_path, chain_kwargs)
        log_weights = iutils.get_kappa_log_weights(k_bnn[idx, :],
                                                   log_p_k_given_omega_int[idx, :])
        np.save(path, log_weights)
        return log_weights

    def get_kappa_log_weights_grid(self, idx,
                                   grid=None,
                                   n_samples=None,
                                   n_mc_dropout=None,
                                   interim_pdf_func=None):
        """Get log weights for reweighted kappa posterior, analytically
        on a grid

        Parameters
        ----------
        idx : int
            Index of sightline in test set
        grid : np.ndarray, optional
            Grid of kappa values at which to evaluate log weights
            (May be overridden with what was used previously, if
            kappa samples were already drawn and stored)
        n_samples : int, optional
            Number of samples per dropout, for getting kappa samples.
            (May be overridden with what was used previously, if
            kappa samples were already drawn and stored)
        n_mc_dropout : int, optional
            Number of dropout iterates, for getting kappa samples.
            (May be overridden with what was used previously, if
            kappa samples were already drawn and stored)
        interim_pdf_func : callable, optional
            Function that returns the density of the interim prior

        Note
        ----
        log doesn't help with numerical stability since we divide
        probabilities directly, but we're keeping this just for
        consistency

        Returns
        -------
        np.ndarray
            kappa grid, log weights for each of the BNN samples for
            this sightline
        """
        os.makedirs(self.reweighted_grid_dir, exist_ok=True)
        path = osp.join(self.reweighted_grid_dir,
                        f'log_weights_{idx}.npy')
        if osp.exists(path):
            return np.load(path)
        # Get unflattened, i.e. [n_test, 1, n_mc_dropout, n_samples]
        k_bnn = self.get_bnn_kappa(n_samples=n_samples,
                                   n_mc_dropout=n_mc_dropout,
                                   flatten=False)
        k_bnn = k_bnn[idx, 0, :, :]  # [n_mc_dropout, n_samples]
        n_mc_dropout, n_samples = k_bnn.shape
        numer = np.zeros(grid.shape)  # init numerator
        # Fit a normal for each MC dropout
        for d in range(n_mc_dropout):
            samples_d = k_bnn[d, :]
            norm_d = scipy.stats.norm(loc=samples_d.mean(),
                                      scale=samples_d.std())
            bnn_prob_d = norm_d.pdf(grid)
            numer += (bnn_prob_d - numer)/(d+1)  # running mean
        np.save(osp.join(self.out_dir, f'grid_bnn_gmm_{idx}.npy'),
                numer)
        denom = interim_pdf_func(grid)
        log_weights = np.log(numer/denom)
        log_weights_grid = np.stack([grid, log_weights], axis=0)
        np.save(path, log_weights_grid)
        return log_weights_grid

    def get_reweighted_bnn_kappa(self, n_resamples, grid_kappa_kwargs,
                                 ):
        if osp.exists(self.reweighted_bnn_kappa_grid_path):
            print("Reading existing reweighted BNN kappa...")
            return np.load(self.reweighted_bnn_kappa_grid_path)
        n_test = len(self.test_dataset)
        k_reweighted = np.empty([n_test, 1, n_resamples])
        for idx in tqdm(range(n_test), desc='evaluating, resampling'):
            # On a grid
            grid, log_p = self.get_kappa_log_weights_grid(idx,
                                                          **grid_kappa_kwargs)
            resamples = iutils.resample_from_pdf(grid, log_p, n_resamples)
            k_reweighted[idx, 0, :] = resamples
            # Per sample
            log_p_sample = self.get_kappa_log_weights(idx, **grid_kappa_kwargs)
        # Grid resamples for all sightlines
        np.save(self.reweighted_bnn_kappa_grid_path, k_reweighted)
        # Per-sample resamples for all sightlines

        return k_reweighted

    def visualize_omega_post(self, chain_path, chain_kwargs,
                             corner_kwargs, log_idx=None):
        # MCMC samples ~ [n_omega, 2]
        omega_post_samples = iutils.get_mcmc_samples(chain_path, chain_kwargs)
        if log_idx is not None:
            omega_post_samples[:, log_idx] = np.exp(omega_post_samples[:, log_idx])
        fig = corner.corner(omega_post_samples,
                            **corner_kwargs)

        fig.savefig(osp.join(self.out_dir, 'omega_post.pdf'))

    def visualize_kappa_post(self, idx, n_samples, n_mc_dropout,
                             interim_pdf_func, grid=None):
        log_weights = self.get_kappa_log_weights(idx,
                                                 n_samples,
                                                 n_mc_dropout,
                                                 interim_pdf_func)  # [n_samples]
        grid, log_w_grid = self.get_kappa_log_weights_grid(idx,
                                                           grid,
                                                           n_samples,
                                                           n_mc_dropout,
                                                           interim_pdf_func)
        w_grid = np.exp(log_w_grid)
        k_bnn = self.get_bnn_kappa(n_samples=n_samples,
                                   n_mc_dropout=n_mc_dropout)  # [n_test, n_samples]
        true_k = self.get_true_kappa(is_train=False)
        fig, ax = plt.subplots()
        # Original posterior
        bins = np.histogram_bin_edges(k_bnn[idx].squeeze(), bins='scott',)
        ax.hist(k_bnn[idx].squeeze(),
                histtype='step',
                bins=bins,
                density=True,
                color='#8ca252',
                label='original')
        # Reweighted posterior, per sample
        ax.hist(k_bnn[idx].squeeze(),
                histtype='step',
                bins=25,
                density=True,
                weights=np.exp(log_weights),
                color='#d6616b',
                label='reweighted per sample')
        # Reweighted posterior, analytical
        reweighted_k_bnn = self.get_reweighted_bnn_kappa(None, None)[idx, 0, :]
        bin_vals, bin_edges = np.histogram(reweighted_k_bnn, bins='scott',
                                           density=True)
        norm_factor = np.max(bin_vals)/np.max(w_grid)
        ax.plot(grid, norm_factor*w_grid,
                color='#d6616b',
                label='reweighted on grid')
        # Truth
        ax.axvline(true_k[idx].squeeze(), color='k', label='truth')
        ax.set_xlabel(r'$\kappa$')
        ax.legend()

    @property
    def pre_reweighting_metrics_path(self):
        return osp.join(self.out_dir, 'pre_metrics.csv')

    @property
    def pre_reweighting_metrics(self):
        return pd.read_csv(self.pre_reweighting_metrics_path,
                           index_col=False)

    @property
    def post_reweighting_metrics_path(self):
        return osp.join(self.out_dir, 'post_metrics.csv')

    @property
    def post_reweighting_metrics(self):
        return pd.read_csv(self.post_reweighting_metrics_path,
                           index_col=False)

    def compute_metrics(self):
        """Evaluate metrics for model selection
        """
        columns = ['minus_sig', 'med', 'plus_sig']
        columns += ['log_p', 'mad', 'mae']
        # mae = median absolute errors, robust measure of accuracy
        # mad = median absolute deviation, robust measure of precision
        # Metrics on pre-reweighting BNN posteriors
        k_bnn_pre = self.get_bnn_kappa()
        pre_metrics = pd.DataFrame(columns=columns)
        # Metrics on post-reweighting BNN posteriors
        k_bnn_post = self.get_reweighted_bnn_kappa(None, None)
        post_metrics = pd.DataFrame(columns=columns)
        # True kappa
        k_test = self.get_true_kappa(is_train=False).squeeze()
        n_test = len(k_test)
        for i in range(n_test):
            # Init rows to append
            pre_stats = dict()
            post_stats = dict()
            # Slice samples for this sightline
            pre_samples = k_bnn_pre[i, 0, :]
            post_samples = k_bnn_post[i, 0, :]
            # Evaluate log p at truth
            true_k = k_test[i]
            pre_log_p = np.log(np.load(osp.join(self.out_dir,
                                                f'grid_bnn_gmm_{i}.npy')))
            grid, post_log_p = self.get_kappa_log_weights_grid(i)
            # Make finer resolution
            fine_grid = np.linspace(-0.2, 0.2, 10000)
            fine_pre_log_p = np.interp(fine_grid, grid, pre_log_p)
            fine_post_log_p = np.interp(fine_grid, grid, post_log_p)
            # Evaluate fine log p at truth
            true_i = np.argmin(np.abs(fine_grid - true_k))
            pre_stats.update(log_p=fine_pre_log_p[true_i])
            post_stats.update(log_p=fine_post_log_p[true_i])
            # Compute descriptive stats
            lower, med, upper = np.quantile(pre_samples,
                                            [0.5-0.34, 0.5, 0.5+0.34])
            pre_stats.update(minus_sig=med - lower,
                             med=med,
                             plus_sig=upper - med,
                             mae=np.median(np.abs(pre_samples - true_k)),
                             mad=scipy.stats.median_abs_deviation(pre_samples))
            lower, med, upper = np.quantile(post_samples,
                                            [0.5-0.34, 0.5, 0.5+0.34])
            post_stats.update(minus_sig=med - lower,
                              med=med,
                              plus_sig=upper - med,
                              mae=np.median(np.abs(post_samples - true_k)),
                              mad=scipy.stats.median_abs_deviation(post_samples))
            pre_metrics = pre_metrics.append(pre_stats,
                                             ignore_index=True)
            post_metrics = post_metrics.append(post_stats,
                                               ignore_index=True)
        # Evaluate average metrics over entire test set
        pre_metrics = pre_metrics.append(pre_metrics.mean(),
                                         ignore_index=True)
        pre_metrics = pre_metrics.append(pre_metrics.median(),
                                         ignore_index=True)
        post_metrics = post_metrics.append(post_metrics.mean(),
                                           ignore_index=True)
        post_metrics = post_metrics.append(post_metrics.median(),
                                           ignore_index=True)
        # Save as CSV
        pre_metrics.to_csv(self.pre_reweighting_metrics_path,
                           index=False)
        post_metrics.to_csv(self.post_reweighting_metrics_path,
                            index=False)

    def get_calibration_plot(self, k_bnn):
        """Plot calibration (should be run on the validation set)

        Parameters
        ----------
        k_bnn : np.ndarray
            Reweighted BNN samples, of shape [n_test, Y_dim, n_samples]
        """
        k_bnn = np.transpose(k_bnn, [2, 0, 1])  # [n_samples, n_test, Y_dim=1]
        y_mean = np.mean(k_bnn, axis=0)
        k_val = self.get_true_kappa(is_train=False)
        train_cov = self.Y_std.cpu().numpy()

        fig = calib.plot_calibration(post_samples=k_bnn,
                                     y_mean=y_mean,
                                     y_truth=k_val,
                                     cov=train_cov,
                                     show_plot=False,
                                     ls='--',
                                     color_map=['tab:gray', '#880519'],
                                     legend=['Perfect calibration',
                                             'Dropout'])
        fig.savefig(osp.join(self.out_dir, 'calibration.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=200)

    # TODO: add docstring
    # TODO: implement initialization from PSO
    # TODO: implement method `visualize_kappa_post_all` comparing before vs after
    # for all sightlines in test set
    # TODO: implement method `visualize_learned_prior` stacking predictions
    # for all sightlines in prior
    # TODO: add markdown to notebook


