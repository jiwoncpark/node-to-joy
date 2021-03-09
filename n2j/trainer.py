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
import n2j.losses.nll as losses
import n2j.models.gnn as gnn
from n2j.trainval_data.trainval_data_utils import Standardizer, Slicer


def get_idx(orig_list, sub_list):
    idx = []
    for item in sub_list:
        idx.append(orig_list.index(item))
    return idx


def is_decreasing(arr):
    """Returns True if array is strictly monotonically decreasing

    """
    return np.all(np.diff(arr) < 0.0)


class Trainer:
    Y_def = {0: 'k', 1: 'g1', 2: 'g2'}

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

    def seed_everything(self):
        """Seed the training and sampling for reproducibility

        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_dataset(self, features, raytracing_out_dir, healpix, n_data,
                     is_train, batch_size, aperture_size,
                     stop_mean_std_early=False, sub_features=None):
        self.batch_size = batch_size
        self.X_dim = len(features) if sub_features is None else len(sub_features)
        self.Y_dim = 3
        dataset = CosmoDC2Graph(healpix=healpix,
                                raytracing_out_dir=raytracing_out_dir,
                                aperture_size=aperture_size,
                                n_data=n_data,
                                features=features,
                                stop_mean_std_early=stop_mean_std_early,
                                )
        if is_train:
            self.train_dataset = dataset
            stats = self.train_dataset.data_stats
            if sub_features is not None:
                idx = get_idx(features, sub_features)
                slicing = Slicer(idx)
                norming = Standardizer(stats['X_mean'][:, idx],
                                       stats['X_std'][:, idx])
                self.transform_X = transforms.Compose([slicing, norming])

            else:
                self.transform_X = Standardizer(stats['X_mean'],
                                                stats['X_std'])
            self.transform_Y = Standardizer(stats['Y_mean'],
                                            stats['Y_std'])
            self.train_dataset.transform_X = self.transform_X
            self.train_dataset.transform_Y = self.transform_Y
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=True)
        else:
            self.val_dataset = dataset
            self.val_dataset.transform_X = self.transform_X
            self.val_dataset.transform_Y = self.transform_Y
            self.val_loader = DataLoader(self.val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         drop_last=True)

    def configure_loss_fn(self, loss_type):
        nll_obj = getattr(losses, loss_type)(Y_dim=self.Y_dim,
                                             device=self.device)
        self.nll_obj = nll_obj
        self.nll_type = self.nll_obj.__class__.__name__
        self.out_dim = nll_obj.out_dim

    def configure_model(self, model_name, model_kwargs={}):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = getattr(gnn, model_name)(in_channels=self.X_dim,
                                              out_channels=self.out_dim,
                                              **self.model_kwargs)
        self.model.to(self.device)

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
        model_fname = '{:s}_{:s}_{:s}.mdl'.format(self.nll_type,
                                                  self.model_name,
                                                  time_stamp)
        self.model_path = os.path.join(self.checkpoint_dir, model_fname)
        print(self.model_path)
        torch.save(state, self.model_path)

    def configure_optim(self,
                        optim_kwargs={},
                        lr_scheduler_kwargs={'factor': 0.5, 'min_lr': 1.e-7}):
        """Configure optimization-related objects

        """
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimizer = optim.Adam(self.model.parameters(), **self.optim_kwargs)
        self.optimizer.zero_grad()
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                 **self.lr_scheduler_kwargs)

    def train_single_epoch(self):
        train_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            out = self.model(batch)
            loss = self.nll_obj(out, batch.y)
            loss.backward()
            self.optimizer.step()
            train_loss += (loss.detach().cpu().item() - train_loss)/(1.0+i)
        return train_loss

    def train(self, n_epochs, sample_kwargs={}):
        self.model.train()
        # Training loop
        self.n_epochs = n_epochs
        progress = tqdm(range(self.epoch, self.n_epochs))
        for epoch_i in progress:
            train_loss_i = self.train_single_epoch()
            val_loss_i = self.infer()
            self.logger.add_scalars('metrics/loss',
                                    dict(train=train_loss_i, val=val_loss_i),
                                    epoch_i)
            self.eval_posterior(epoch_i, **sample_kwargs)
            self.epoch = epoch_i
            # Stop early if val loss doesn't decrease for 5 consecutive epochs
            self.early_stop_crit.append(val_loss_i)
            self.early_stop_crit = self.early_stop_crit[-5:]
            if ~is_decreasing(self.early_stop_crit) and (epoch_i > 30):
                break
            if val_loss_i < self.last_saved_val_loss:
                os.remove(self.model_path) if os.path.exists(self.model_path) else None
                self.save_state(train_loss_i, val_loss_i)
                self.last_saved_val_loss = val_loss_i
        self.logger.close()

    def infer(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.nll_obj(out, batch.y)
                val_loss += (loss.cpu().item() - val_loss)/(1.0+i)
        self.lr_scheduler.step(val_loss)
        return val_loss

    def eval_posterior(self, epoch_i, n_samples=200, n_mc_dropout=20):
        # Fetch precomputed Y_mean, Y_std to de-standardize samples
        Y_mean = self.train_dataset.data_stats['Y_mean'].to(self.device)
        Y_std = self.train_dataset.data_stats['Y_std'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            samples = np.empty([self.batch_size,
                               n_mc_dropout,
                               n_samples,
                               self.Y_dim])
            for i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                # Get ground truth
                y_val = (batch.y*Y_std + Y_mean).cpu().numpy()
                for mc_iter in range(n_mc_dropout):
                    out = self.model(batch)
                    # Get pred samples
                    self.nll_obj.set_trained_pred(out)
                    if 'Double' in self.nll_type:
                        self.logger.add_histogram('w2', self.nll_obj.w2, epoch_i)
                    mc_samples = self.nll_obj.sample(Y_mean,
                                                     Y_std,
                                                     n_samples,
                                                     sample_seed=self.seed)
                    samples[:, mc_iter, :, :] = mc_samples
                break  # only process the first batch
        self.log_metrics(epoch_i, samples, y_val)

    def log_metrics(self, epoch_i, samples, y_val):
        # Log metrics on pred
        samples = samples.transpose(0, 3, 1, 2).reshape([self.batch_size,
                                                        self.Y_dim,
                                                        -1])
        mean_pred = np.mean(samples, axis=-1)  # [batch_size, Y_dim]
        std_pred = np.std(samples, axis=-1)
        prec = np.abs(std_pred)
        err = np.abs((mean_pred - y_val))
        z = ((mean_pred - y_val)/(std_pred + 1.e-7))
        for i, name in self.Y_def.items():
            self.logger.add_scalars('metrics/{:s}'.format(name),
                                    dict(med_ae=np.median(err, axis=0)[i],
                                         med_prec=np.median(prec, axis=0)[i]),
                                    epoch_i)
            self.logger.add_histogram('absolute precision/{:s}'.format(name),
                                      prec[:, i],
                                      epoch_i)
            self.logger.add_histogram('absolute error/{:s}'.format(name),
                                      err[:, i],
                                      epoch_i)
            self.logger.add_histogram('z/{:s}'.format(name),
                                      z[:, i],
                                      epoch_i)

    def __repr__(self):
        keys = ['X_dim', 'features', 'sub_features', 'Y_dim', 'out_dim']
        keys += ['batch_size', 'epoch', 'n_epochs']
        vals = [getattr(self, k) for k in keys if hasattr(self, k)]
        metadata = dict(zip(keys, vals))
        if hasattr(self, 'model_kwargs'):
            metadata.update(self.model_kwargs)
        if hasattr(self, 'optim_kwargs'):
            metadata.update(self.optim_kwargs)
        if hasattr(self, 'lr_scheduler_kwargs'):
            metadata.update(self.lr_scheduler_kwargs)
        if hasattr(self, 'nll_obj'):
            metadata.update({'nll_type': self.nll_type})
        return json.dumps(metadata)
