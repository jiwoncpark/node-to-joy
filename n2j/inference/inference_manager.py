"""Class managing the model inference

"""
import os
import os.path as osp
import random
import datetime
import json
import numpy as np
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
from n2j.trainval_data.utils.transform_utils import (Standardizer,
                                                     Slicer,
                                                     MagErrorSimulatorTorch,
                                                     get_bands_in_x)
import n2j.inference.summary_stats_baseline as ssb
import n2j.inference.calibration as calib


def get_idx(orig_list, sub_list):
    idx = []
    for item in sub_list:
        idx.append(orig_list.index(item))
    return idx


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
                     noise_kwargs=dict(mag=dict(
                                                override_kwargs=None,
                                                depth=5,
                                                )
                                       )):
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
        dataset = CosmoDC2Graph(**data_kwargs)
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
            if sub_features:
                idx = get_idx(features, sub_features)
                self.X_mean = stats['X_mean'][:, idx]
                self.X_std = stats['X_std'][:, idx]
                slicing = Slicer(idx)
                mag_idx, which_bands = get_bands_in_x(sub_features)
                print(f"Mag errors added to {which_bands}")
                magerr = MagErrorSimulatorTorch(mag_idx=mag_idx,
                                                which_bands=which_bands,
                                                **noise_kwargs['mag'])
                norming = Standardizer(self.X_mean, self.X_std)
                self.transform_X = transforms.Compose([slicing,
                                                      magerr,
                                                      norming])
            else:
                self.X_mean = stats['X_mean']
                self.X_std = stats['X_std']
                self.transform_X = Standardizer(self.X_mean, self.X_std)
            # Transforming global Y
            if sub_target:
                idx_Y = get_idx(target, sub_target)
                self.Y_mean = stats['Y_mean'][:, idx_Y]
                self.Y_std = stats['Y_std'][:, idx_Y]
                slicing_Y = Slicer(idx_Y)
                norming_Y = Standardizer(self.Y_mean, self.Y_std)
                self.transform_Y = transforms.Compose([slicing_Y, norming_Y])
            else:
                self.transform_Y = Standardizer(self.Y_mean, self.Y_std)
            # Transforming local Y
            if sub_target_local:
                idx_Y_local = get_idx(target_local, sub_target_local)
                self.Y_local_mean = stats['Y_local_mean'][:, idx_Y_local]
                self.Y_local_std = stats['Y_local_std'][:, idx_Y_local]
                slicing_Y_local = Slicer(idx_Y_local)
                norming_Y_local = Standardizer(self.Y_local_mean,
                                               self.Y_local_std)
                self.transform_Y_local = transforms.Compose([slicing_Y_local,
                                                            norming_Y_local])
            else:
                self.transform_Y_local = Standardizer(self.Y_local_mean,
                                                      self.Y_local_std)
            self.train_dataset.transform_X = self.transform_X
            self.train_dataset.transform_Y = self.transform_Y
            self.train_dataset.transform_Y_local = self.transform_Y_local
            # Loading option 1: Subsample from a distribution
            if data_kwargs['subsample_pdf_func'] is not None:
                self.class_weight = None
                train_subset = torch.utils.data.Subset(self.train_dataset,
                                                       stats['subsample_idx'])
                self.train_dataset = train_subset
                self.train_loader = DataLoader(self.train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               drop_last=False)
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
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   drop_last=True)
            print(f"Train dataset size: {len(self.train_dataset)}")
        #################
        # Test (or val) #
        #################
        else:
            self.test_dataset = dataset
            # Compute or retrieve stats necessary for resampling
            # before setting any kind of transforms
            # Note: stats_test.pt is in our_dir, not checkpoint_dir
            if data_kwargs['subsample_pdf_func'] is not None:
                stats_test_path = osp.join(self.out_dir, 'stats_test.pt')
                if osp.exists(stats_test_path):
                    stats_test = torch.load(stats_test_path)
                else:
                    stats_test = self.test_dataset.data_stats_valtest
                    torch.save(stats_test, stats_test_path)
            self.test_dataset.transform_X = self.transform_X
            self.test_dataset.transform_Y = self.transform_Y
            self.test_dataset.transform_Y_local = self.transform_Y_local
            # Test loading option 1: Subsample from a distribution
            if data_kwargs['subsample_pdf_func'] is not None:
                self.class_weight = None
                test_subset = torch.utils.data.Subset(self.test_dataset,
                                                      stats_test['subsample_idx'])
                self.test_dataset = test_subset
                self.test_loader = DataLoader(self.test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=self.num_workers,
                                              drop_last=False)
            else:
                # Test loading option 2: No special sampling, no shuffle
                self.class_weight = None
                self.test_loader = DataLoader(self.test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=self.num_workers,
                                              drop_last=True)
            print(f"Test dataset size: {len(self.test_dataset)}")

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
        state = torch.load(state_path)
        self.model.load_state_dict(state['model'])
        self.model.to(self.device)
        self.epoch = state['epoch']
        train_loss = state['train_loss']
        val_loss = state['val_loss']
        print("Loaded weights at {:s}".format(state_path))
        print("Epoch [{}]: TRAIN Loss: {:.4f}".format(self.epoch, train_loss))
        print("Epoch [{}]: VALID Loss: {:.4f}".format(self.epoch, val_loss))
        self.last_saved_val_loss = val_loss

    def get_bnn_kappa(self, n_samples=50, n_mc_dropout=20):
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
        path = osp.join(self.out_dir, 'k_bnn.npy')
        if osp.exists(path):
            samples = np.load(path)
            return samples
        # Fetch precomputed Y_mean, Y_std to de-standardize samples
        Y_mean = self.Y_mean.to(self.device)
        Y_std = self.Y_std.to(self.device)
        n_test = len(self.test_dataset)
        self.model.eval()
        with torch.no_grad():
            samples = np.empty([n_test, n_mc_dropout, n_samples, self.Y_dim])
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
        samples = samples.transpose(0, 3, 1, 2).reshape([n_test, self.Y_dim, -1])
        np.save(path, samples)
        return samples

    def get_true_kappa(self, is_train, add_suffix='',
                       compute_summary=True, save=True):
        """Fetch true kappa (for train/val/test)

        Parameters
        ----------
        is_train : bool
            Whether to get true kappas for train (test otherwise)
        add_suffix : str, optional
            Suffix to append to the filename. Useful in case there are
            variations in the test distribution
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
            suffix = 'train'
            n_data = len(self.train_dataset)
        else:
            loader = self.test_loader
            suffix = 'test'
            n_data = len(self.test_dataset)
        path = osp.join(self.out_dir, f'k_{suffix}{add_suffix}.npy')
        ss_path = osp.join(self.out_dir,
                           f'summary_stats_{suffix}.npy')
        if osp.exists(path):
            if compute_summary and osp.exists(ss_path):
                true_kappa = np.load(path)
                return true_kappa
        print(f"Saving 'k_{suffix}{add_suffix}.npy'...")
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
        train_ss_obj.set_stats(osp.join(self.out_dir,
                               'summary_stats_train.npy'))
        test_ss_obj = ssb.SummaryStats(len(self.test_dataset),
                                       pos_indices)
        test_ss_obj.set_stats(osp.join(self.out_dir,
                              'summary_stats_test.npy'))
        matcher = ssb.Matcher(train_ss_obj, test_ss_obj,
                              train_k,
                              osp.join(self.out_dir, 'matching'),
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
        path = osp.join(self.out_dir, 'log_p_k_given_omega_int.npy')
        if osp.exists(path):
            return np.load(path)
        k_train = self.get_true_kappa(is_train=True).squeeze(1)
        k_bnn = self.get_bnn_kappa(n_samples=n_samples,
                                   n_mc_dropout=n_mc_dropout).squeeze(1)
        log_p_k_given_omega_int = iutils.get_log_p_k_given_omega_int_analytic(k_train=k_train,
                                                                              k_bnn=k_bnn,
                                                                              interim_pdf_func=interim_pdf_func)
        np.save(path, log_p_k_given_omega_int)
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
                              chain_path, chain_kwargs,
                              interim_pdf_func):
        path = osp.join(self.out_dir, f'log_weights_{idx}.npy')
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
                             chain_path, chain_kwargs, interim_pdf_func):
        log_weights = self.get_kappa_log_weights(idx,
                                                 n_samples,
                                                 n_mc_dropout,
                                                 chain_path,
                                                 chain_kwargs,
                                                 interim_pdf_func)  # [n_samples]
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
        # Reweighted posterior
        ax.hist(k_bnn[idx].squeeze(),
                histtype='step',
                bins=25,
                density=True,
                weights=np.exp(log_weights),
                color='#d6616b',
                label='reweighted')
        # Truth
        ax.axvline(true_k[idx].squeeze(), color='k', label='truth')
        ax.set_xlabel(r'$\kappa$')
        ax.legend()

    def get_calibration_plot(self):
        k_bnn = self.get_bnn_kappa()
        k_bnn = np.transpose(k_bnn, [2, 0, 1])  # [n_samples, n_sightlines, 1]
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


