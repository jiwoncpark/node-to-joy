import os
import random
import datetime
import json
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch_geometric.data import DataLoader
from n2j.trainval_data.graphs.cosmodc2_graph import CosmoDC2Graph
import n2j.losses as losses
import n2j.models as models
from n2j.trainval_data.utils.transform_utils import Standardizer, Slicer


def get_idx(orig_list, sub_list):
    idx = []
    for item in sub_list:
        idx.append(orig_list.index(item))
    return idx


def is_decreasing(arr):
    """Returns True if array ever decreased

    """
    return np.any(np.diff(arr) < 0.0)


class Trainer:

    def __init__(self, device_type, checkpoint_dir='trained_models', seed=123):
        self.device = torch.device(device_type)
        self.seed = seed
        self.seed_everything()
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'runs'))
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
                     sub_features=None, sub_target=None, sub_target_local=None):
        self.batch_size = batch_size
        # X metadata
        features = data_kwargs['features']
        self.sub_features = sub_features if sub_features else features
        self.X_dim = len(self.sub_features)
        # Global y metadata
        target = ['final_kappa', 'final_gamma1', 'final_gamma2']
        self.sub_target = sub_target if sub_target else target
        self.Y_dim = len(self.sub_target)
        # Lobal y metadata
        target_local = ['halo_mass', 'redshift']
        self.sub_target_local = sub_target_local if sub_target_local else target_local
        self.Y_local_dim = len(self.sub_target_local)
        dataset = CosmoDC2Graph(**data_kwargs)
        if is_train:
            self.train_dataset = dataset
            stats = self.train_dataset.data_stats
            # Transforming X
            if sub_features:
                idx = get_idx(features, sub_features)
                self.X_mean = stats['X_mean'][:, idx]
                self.X_std = stats['X_std'][:, idx]
                slicing = Slicer(idx)
                norming = Standardizer(self.X_mean, self.X_std)
                self.transform_X = transforms.Compose([slicing, norming])
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
                norming_Y_local = Standardizer(self.Y_local_mean, self.Y_local_std)
                self.transform_Y_local = transforms.Compose([slicing_Y_local,
                                                            norming_Y_local])
            else:
                self.transform_Y_local = Standardizer(self.Y_local_mean, self.Y_local_std)
            self.train_dataset.transform_X = self.transform_X
            self.train_dataset.transform_Y = self.transform_Y
            self.train_dataset.transform_Y_local = self.transform_Y_local
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=True)
        else:
            self.val_dataset = dataset
            self.val_dataset.transform_X = self.transform_X
            self.val_dataset.transform_Y = self.transform_Y
            self.val_dataset.transform_Y_local = self.transform_Y_local
            self.val_loader = DataLoader(self.val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         drop_last=True)

    def configure_loss_fn(self, loss_type):
        loss_obj = getattr(losses, loss_type)()
        self.loss_obj = loss_obj
        self.loss_type = self.loss_obj.__class__.__name__

    def configure_model(self, model_name, model_kwargs={}):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = getattr(models, model_name)(**self.model_kwargs)
        self.model.to(self.device)
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
        model_fname = '{:s}_{:s}_{:s}.mdl'.format(self.loss_type,
                                                  self.model_name,
                                                  time_stamp)
        self.model_path = os.path.join(self.checkpoint_dir, model_fname)
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
            batch = batch.to(self.device)
            x, u = self.model(batch)
            loss_local, loss_global = self.model.loss(x, u, batch)
            loss = self.weight_local_loss*loss_local + loss_global
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += (loss.detach().cpu().item() - train_loss)/(1.0+i)
            self.logger.add_scalar('metrics/iter_loss', train_loss,
                                   epoch_i*n_batches + i)
        return train_loss

    def train(self, n_epochs, eval_every=1, eval_on_train=False, sample_kwargs={}):
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
            if False:
                if (epoch_i+1) % eval_every == 0:
                    if eval_on_train:
                        self.eval_posterior(epoch_i, **sample_kwargs, on_train=True)
                    self.eval_posterior(epoch_i, **sample_kwargs, on_train=False)
            self.epoch = epoch_i
            # Stop early if val loss doesn't decrease for 10 consecutive epochs
            self.early_stop_crit.append(val_loss_i)
            self.early_stop_crit = self.early_stop_crit[-self.early_stop_memory:]
            memory_filled = len(self.early_stop_crit) == self.early_stop_memory
            if ~is_decreasing(self.early_stop_crit) and memory_filled:
                break
            if val_loss_i < self.last_saved_val_loss:
                os.remove(self.model_path) if os.path.exists(self.model_path) else None
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
        return val_loss

    def eval_posterior(self, epoch_i, n_samples=200, n_mc_dropout=20,
                       on_train=False):
        # Fetch precomputed Y_mean, Y_std to de-standardize samples
        Y_mean = self.Y_mean.to(self.device)
        Y_std = self.Y_std.to(self.device)
        if on_train:
            loader = self.train_loader
            prefix = 'train'
            n_mc_dropout = 1
            n_samples = 1
        else:
            loader = self.val_loader
            prefix = 'val'
        B = self.batch_size  # for convenience
        n_data = len(loader)*B
        self.model.eval()
        with torch.no_grad():
            samples = np.empty([n_data,
                               n_mc_dropout,
                               n_samples,
                               self.Y_dim])
            y_unnormed = np.empty([n_data, self.Y_dim])
            edge_index_list = []
            w_list = []
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                # Get ground truth
                y_unnormed[i*B: (i+1)*B, :] = (batch.y*Y_std + Y_mean).cpu().numpy()
                for mc_iter in range(n_mc_dropout):
                    out, (edge_index, w) = self.model(batch)
                    # Get pred samples
                    self.loss_obj.set_trained_pred(out)
                    if 'Double' in self.loss_type:
                        self.logger.add_histogram('{:s}/w2'.format(prefix),
                                                  self.loss_obj.w2, epoch_i)
                        mc_samples = self.loss_obj.sample(Y_mean,
                                                          Y_std,
                                                          n_samples,
                                                          sample_seed=self.seed)
                    samples[i*B: (i+1)*B, mc_iter, :, :] = mc_samples
                edge_index_list.append(edge_index.detach().cpu().numpy())
                w_list.append(w.detach().cpu().numpy())
        samples = samples.transpose(0, 3, 1, 2).reshape([n_data, self.Y_dim, -1])
        self.log_metrics(epoch_i, samples, y_unnormed, prefix)
        summary = dict(samples=samples,
                       y_val=y_unnormed,
                       batch=batch.batch.detach().cpu().numpy(),
                       edge_index=np.concatenate(edge_index_list, axis=1),
                       w=np.concatenate(w_list, axis=0)
                       )
        return summary

    def log_metrics(self, epoch_i, samples, y_val, prefix):
        # Log metrics on pred
        mean_pred = np.mean(samples, axis=-1)  # [batch_size, Y_dim]
        std_pred = np.std(samples, axis=-1)
        prec = np.abs(std_pred)
        err = np.abs((mean_pred - y_val))
        z = ((mean_pred - y_val)/(std_pred + 1.e-7))
        for i, name in enumerate(self.sub_target):
            self.logger.add_scalars('{:s}/metrics/{:s}'.format(prefix, name),
                                    dict(med_ae=np.median(err, axis=0)[i],
                                         med_prec=np.median(prec, axis=0)[i]),
                                    epoch_i)
            self.logger.add_histogram('{:s}/prec/{:s}'.format(prefix, name),
                                      prec[:, i],
                                      epoch_i)
            self.logger.add_histogram('{:s}/MAE/{:s}'.format(prefix, name),
                                      err[:, i],
                                      epoch_i)
            self.logger.add_histogram('{:s}/z/{:s}'.format(prefix, name),
                                      z[:, i],
                                      epoch_i)

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
        if hasattr(self, 'loss_obj'):
            metadata.update({'loss_type': self.loss_type})
        return json.dumps(metadata)
