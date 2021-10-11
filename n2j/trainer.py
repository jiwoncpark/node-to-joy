"""Class managing the model training

"""
import os
import os.path as osp
import random
import datetime
import json
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.data import DataLoader
from n2j.trainval_data.graphs.cosmodc2_graph import CosmoDC2Graph
import n2j.models as models
from n2j.trainval_data.utils.transform_utils import (ComposeXYLocal,
                                                     Standardizer,
                                                     Slicer,
                                                     MagErrorSimulatorTorch,
                                                     Rejector,
                                                     get_bands_in_x,
                                                     get_idx)
import matplotlib.pyplot as plt


def is_decreasing(arr):
    """Returns True if array ever decreased

    """
    return np.any(np.diff(arr) < 0.0)


class Trainer:

    def __init__(self, device_type, checkpoint_dir='trained_models', seed=123):
        self.device_type = device_type
        self.device = torch.device(self.device_type)
        self.seed = seed
        self.seed_everything()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = SummaryWriter(osp.join(self.checkpoint_dir, 'runs'))
        self.epoch = 0
        self.early_stop_crit = []
        self.last_saved_val_loss = np.inf
        self.model_path = 'dummy_path_name'
        # Any non-weight variables of the model to log
        self.model_log = {}

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
            norming_X_meta = Standardizer(stats['X_meta_mean'],
                                          stats['X_meta_std'])
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
                                                      [norming_Y_local],
                                                      [norming_X_meta])
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
        ##############
        # Validation #
        ##############
        else:
            self.val_dataset = dataset
            # Compute or retrieve stats necessary for resampling
            # before setting any kind of transforms
            if data_kwargs['subsample_pdf_func'] is not None:
                stats_val_path = osp.join(self.checkpoint_dir, 'stats_val.pt')
                if osp.exists(stats_val_path):
                    stats_val = torch.load(stats_val_path)
                else:
                    stats_val = self.val_dataset.data_stats_valtest
                    torch.save(stats_val, stats_val_path)
            self.val_dataset.transform_X_Y_local = self.transform_X_Y_local
            self.val_dataset.transform_Y = self.transform_Y
            # Val loading option 1: Subsample from a distribution
            if data_kwargs['subsample_pdf_func'] is not None:
                self.class_weight = None
                val_subset = torch.utils.data.Subset(self.val_dataset,
                                                     stats_val['subsample_idx'])
                self.val_dataset = val_subset
                self.val_loader = DataLoader(self.val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=self.num_workers,
                                             drop_last=False)
            else:
                # Val loading option 2: No special sampling, no shuffle
                self.class_weight = None
                self.val_loader = DataLoader(self.val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=self.num_workers,
                                             drop_last=True)
            print(f"Val dataset size: {len(self.val_dataset)}")

    def configure_model(self, model_name, model_kwargs={}):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = getattr(models, model_name)(**self.model_kwargs)
        self.model.to(self.device)
        if self.class_weight is not None:
            self.model.class_weight = self.class_weight.to(self.device)
        print("class weight: ", self.model.class_weight)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of params: {n_params}")

    def load_state(self, state_path):
        """Load the state dict of the past training

        Parameters
        ----------
        state_path : str or osp.object
            path of the state dict to load

        """
        state = torch.load(state_path)
        self.model.load_state_dict(state['model'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        self.epoch = state['epoch']
        train_loss = state['train_loss']
        val_loss = state['val_loss']
        print("Loaded weights at {:s}".format(state_path))
        print("Epoch [{}]: TRAIN Loss: {:.4f}".format(self.epoch, train_loss))
        print("Epoch [{}]: VALID Loss: {:.4f}".format(self.epoch, val_loss))
        self.last_saved_val_loss = val_loss

    def save_state(self, train_loss, val_loss):
        """Save the state dict of the current training to disk

        Parameters
        ----------
        train_loss : float
            current training loss
        val_loss : float
            current validation loss

        """
        state = dict(
                 model=self.model.state_dict(),
                 optimizer=self.optimizer.state_dict(),
                 lr_scheduler=self.lr_scheduler.state_dict(),
                 epoch=self.epoch,
                 train_loss=train_loss,
                 val_loss=val_loss,
                 )
        time_fmt = "epoch={:d}_%m-%d-%Y_%H:%M".format(self.epoch)
        time_stamp = datetime.datetime.now().strftime(time_fmt)
        model_fname = '{:s}_{:s}.mdl'.format(self.model_name, time_stamp)
        self.model_path = osp.join(self.checkpoint_dir, model_fname)
        torch.save(state, self.model_path)

    def configure_optim(self, early_stop_memory=20,
                        weight_local_loss=0.1,
                        optim_kwargs={},
                        lr_scheduler_kwargs={'factor': 0.5, 'min_lr': 1.e-7}):
        """Configure optimization-related objects

        """
        self.early_stop_memory = early_stop_memory
        self.weight_local_loss = weight_local_loss
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimizer = optim.Adam(self.model.parameters(), **self.optim_kwargs)
        self.optimizer.zero_grad()
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                 **self.lr_scheduler_kwargs)

    def train_single_epoch(self, epoch_i):
        self.model.train()
        train_loss = 0.0
        n_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            x, u = self.model(batch)
            loss_local, loss_global = self.model.loss(x, u, batch)
            loss = self.weight_local_loss*loss_local + loss_global
            #nan_detected = False
            #for p in self.model.parameters():
            #    if p.grad is None:
            #        continue  # next parameter
            #    if torch.any(torch.isnan(p.grad)):
            #        nan_detected = True
            #        print(nan_detected)
            #if nan_detected:
            #    continue  # next batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.net_out_global.parameters(), 0.01)
            self.optimizer.step()
            train_loss += (loss.detach().cpu().item() - train_loss)/(1.0+i)
            self.logger.add_scalar('metrics/iter_loss', loss.detach().cpu().item(),
                                   epoch_i*n_batches + i)
        return train_loss

    def train(self, n_epochs, sample_kwargs={}):
        self.model.train()
        # Training loop
        self.n_epochs = n_epochs
        progress = tqdm(range(self.epoch, self.n_epochs))
        for epoch_i in progress:
            train_loss_i = self.train_single_epoch(epoch_i)
            val_loss_i = self.infer(epoch_i)
            self.lr_scheduler.step(val_loss_i)
            self.logger.add_scalars('metrics/loss',
                                    dict(train=train_loss_i, val=val_loss_i),
                                    epoch_i)
            self.epoch = epoch_i
            # Stop early if val loss doesn't decrease for 10 consecutive epochs
            self.early_stop_crit.append(val_loss_i)
            self.early_stop_crit = self.early_stop_crit[-self.early_stop_memory:]
            memory_filled = len(self.early_stop_crit) == self.early_stop_memory
            if ~is_decreasing(self.early_stop_crit) and memory_filled:
                break
            if val_loss_i < self.last_saved_val_loss:
                os.remove(self.model_path) if osp.exists(self.model_path) else None
                self.save_state(train_loss_i, val_loss_i)
                self.last_saved_val_loss = val_loss_i
        self.logger.close()

    def infer(self, epoch_i):
        self.model.eval()
        val_loss = 0.0
        total_nll_local = 0.0
        total_nll_global = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                x, u = self.model(batch)
                loss_local, loss_global = self.model.loss(x, u, batch)
                loss = self.weight_local_loss*loss_local + loss_global
                val_loss += (loss.cpu().item() - val_loss)/(1.0+i)
                # Compute metrics
                total_nll_local += (loss_local - total_nll_local)/(1.0+i)  # [1,]
                total_nll_global += (loss_global - total_nll_global)/(1.0+i)  # [1,]
            self.logger.add_scalar('val_nll_local', total_nll_local.item(), epoch_i)
            self.logger.add_scalar('val_nll_kappa', total_nll_global.item(), epoch_i)
            # Make plots on the last batch
            if self.model.global_flow:
                self._log_kappa_recovery_flow(epoch_i, x, u, batch.y)
            else:
                self._log_kappa_recovery(epoch_i, x.cpu(), u.cpu(), batch.y.cpu())
        return val_loss

    def _log_kappa_recovery_flow(self, epoch_i, x, u, y):
        with torch.no_grad():
            u_out, log_det = self.model.net_out_global(u, y)
            log_p = -u_out.pow(2).sum(1)/2
            normed_log_p = log_p + log_det  # [batch_size,]
        self.logger.add_histogram('kappa recovery', normed_log_p, epoch_i)

    def _log_kappa_recovery(self, epoch_i, x, u, y):
        # Convert into mu, sig over normed target
        mu_pred_normed, sig_pred_normed = torch.split(x, len(self.sub_target_local), dim=-1)
        mu_pred_global_normed, sig_pred_global_normed = torch.split(u, 1, dim=-1)
        sig_pred_normed = torch.exp(0.5*sig_pred_normed)
        sig_pred_global_normed = torch.exp(0.5*sig_pred_global_normed)
        # Convert into mu, sig over original target
        mu_global_pred = mu_pred_global_normed*self.Y_std + self.Y_mean
        sig_global_pred = sig_pred_global_normed*self.Y_std
        y = y*self.Y_std + self.Y_mean
        # Convert to numpy
        mu_global_pred = mu_global_pred.squeeze().numpy()
        sig_global_pred = sig_global_pred.squeeze().numpy()
        y = y.squeeze().numpy()
        # Plot
        fig, ax = plt.subplots()
        ax.errorbar(y, y=mu_global_pred, yerr=sig_global_pred,
                    fmt='o', alpha=0.2)
        interval = np.linspace(np.min(y), np.max(y), 20)
        ax.plot(interval, interval, linestyle='--')
        ax.set_xlabel(r"True kappa")
        ax.set_ylabel(r"Pred kappa")
        self.logger.add_figure('kappa recovery', fig, global_step=epoch_i)
        plt.close('all')

    def __repr__(self):
        keys = ['X_dim', 'sub_features', 'sub_target', 'Y_dim', 'out_dim']
        keys += ['batch_size', 'epoch', 'n_epochs']
        keys_vals = [(k, getattr(self, k)) for k in keys if hasattr(self, k)]
        metadata = dict(keys_vals)
        if hasattr(self, 'model_kwargs'):
            metadata.update(self.model_kwargs)
        if hasattr(self, 'optim_kwargs'):
            metadata.update(self.optim_kwargs)
        if hasattr(self, 'lr_scheduler_kwargs'):
            metadata.update(self.lr_scheduler_kwargs)
        return json.dumps(metadata)
