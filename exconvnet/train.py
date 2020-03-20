"""Training various models.
This script trains a model according to the config specifications.

Example
-------
To run this script, pass in the path to the user-defined training config file as the argument::

    $ python ex-con/train.py ex-con/example_user_config.py

"""

import os, sys
import random
import argparse
from addict import Dict
import numpy as np # linear algebra
from tqdm import tqdm
# torch modules
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# exconvnet modules
from exconvnet.trainval_data import XYData
from exconvnet.configs import TrainValConfig
import exconvnet.losses
import exconvnet.models
import exconvnet.inference
import exconvnet.train_utils as train_utils

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Train a model to infer external convergence')
    parser.add_argument('user_cfg_path', help='path to the user-defined training config file')
    args = parser.parse_args()

    return args

def seed_everything(global_seed):
    """Seed everything for reproducibility

    global_seed : int
        seed for `np.random`, `random`, and relevant `torch` backends

    """
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    cfg = TrainValConfig.from_file(args.user_cfg_path)
    # Set device and default data type
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    seed_everything(cfg.global_seed)


    ############
    # Data I/O #
    ############

    # Define training data and loader
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_data = XYData(cfg.data.train_dir, data_cfg=cfg.data)
    train_loader = DataLoader(train_data, batch_size=cfg.optim.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    n_train = train_data.n_data - (train_data.n_data % cfg.optim.batch_size)

    # Define val data and loader
    val_data = XYData(cfg.data.val_dir, data_cfg=cfg.data)
    val_loader = DataLoader(val_data, batch_size=cfg.optim.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    n_val = val_data.n_data - (val_data.n_data % cfg.optim.batch_size)

    if cfg.data.test_dir is not None:
        pass

    #########
    # Model #
    #########

    # Instantiate loss function
    loss_fn = getattr(exconvnet.losses, cfg.model.likelihood_class)(Y_dim=cfg.data.Y_dim, device=device)
    # Instantiate posterior (for logging)
    post = getattr(exconvnet.inference.posterior, loss_fn.posterior_name)(val_data.Y_dim, device, val_data.train_Y_mean, val_data.train_Y_std)
    # Instantiate model
    net = getattr(exconvnet.models, cfg.model.architecture)(num_classes=loss_fn.out_dim)
    net.to(device)

    ################
    # Optimization #
    ################

    # Instantiate optimizer



if __name__ == '__main__':
    main()
