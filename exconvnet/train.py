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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
# exconvnet modules (turn this into relative later)
from trainval_data import ExConvDataset
from configs import TrainValConfig
import losses
import models
import inference
import train_utils as train_utils

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Train a model on existing weak lensing data to infer external convergence')
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

    device = torch.device(cfg.device_type)

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
    dataset = ExConvDataset(x_path=cfg.data.x_path, y_path=cfg.data.y_path, data_cfg=cfg.data)

    # split up the dataset
    train, val = (len(dataset) * np.array(cfg.data.split)[:-1]).astype('int')
    test = len(dataset) - (train + val)
    train, val, test = int(train), int(val), int(test)
    trainset, valset, testset = torch.utils.data.random_split(dataset, [train, val, test])

    # set up data loaders
    train_loader = DataLoader(trainset, batch_size=cfg.optim.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    n_train = len(trainset) - (len(trainset) % cfg.optim.batch_size)

    val_loader = DataLoader(valset, batch_size=cfg.optim.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    n_val = len(valset) - (len(valset) % cfg.optim.batch_size)

    test_loader = DataLoader(testset, batch_size=cfg.optim.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    n_test = len(testset) - (len(testset) % cfg.optim.batch_size)

    #########
    # Model #
    #########

    # Instantiate loss function
    loss_fn = getattr(losses, cfg.model.likelihood_class)()
    # Instantiate posterior (for logging)
    #post = getattr(inference.posterior, loss_fn.posterior_name)(val_data.Y_dim, device, valset.train_Y_mean, valset.train_Y_std)
    # Instantiate model
    net = getattr(models, cfg.model.architecture)(input_size=59, num_classes=1)
    net.to(device)

    ################
    # Optimization #
    ################

    # Instantiate optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.optim.learning_rate, amsgrad=False, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.optim.lr_scheduler.factor, patience=cfg.optim.lr_scheduler.patience, verbose=True)

    # Saving/loading state dicts
    checkpoint_dir = cfg.checkpoint.save_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if cfg.model.load_state:
        epoch, net, optimizer, train_loss, val_loss = train_utils.load_state_dict(cfg.model.state_path, net, optimizer, cfg.optim.n_epochs, device)
        epoch += 1 # resume with next epoch
        last_saved_val_loss = val_loss
        #print(lr_scheduler.state_dict())
    else:
        epoch = 0
        last_saved_val_loss = np.inf

    #logger = SummaryWriter()
    model_path = ''
    print("Training set size: {:d}".format(n_train))
    print("Validation set size: {:d}".format(n_val))
    if cfg.data.test_dir is not None:
        print("Test set size: {:d}".format(n_test))
    progress = tqdm(range(epoch, cfg.optim.n_epochs))
    for epoch in progress:
        print('epoch {}'.format(epoch + 1))
        net.train()
        #net.apply(h0rton.models.deactivate_batchnorm)
        train_loss = 0.0


        for batch_idx, (X_tr, Y_tr) in enumerate(train_loader):
            X_tr = X_tr.to(device)
            Y_tr = Y_tr.to(device)
            # Update weights

            # for debugging
            X_tr = X_tr[0,0].unsqueeze(0).unsqueeze(0)
            Y_tr = torch.Tensor([[0]])

            optimizer.zero_grad()
            pred_tr = net(X_tr)
            loss = loss_fn(pred_tr, Y_tr)
            loss.backward()
            optimizer.step()
            # For logging
            train_loss += (loss.item() - train_loss)/(1 + batch_idx)

        print('training loss: {:.3g}'.format(train_loss))

        '''
        with torch.no_grad():
            net.eval()
            #net.apply(h0rton.models.deactivate_batchnorm)
            val_loss = 0.0
            test_loss = 0.0

            for batch_idx, (X_t, Y_t) in enumerate(test_loader):
                X_t = X_t.to(device)
                Y_t = Y_t.to(device)
                pred_t = net(X_t)
                nograd_loss_t = loss_fn(pred_t, Y_t)
                test_loss += (nograd_loss_t.item() - test_loss)/(1 + batch_idx)

            for batch_idx, (X_v, Y_v) in enumerate(val_loader):
                X_v = X_v.to(device)
                Y_v = Y_v.to(device)
                pred_v = net(X_v)
                nograd_loss_v = loss_fn(pred_v, Y_v)
                val_loss += (nograd_loss_v.item() - val_loss)/(1 + batch_idx)

            tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, train_loss))
            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, val_loss))
            tqdm.write("Epoch [{}/{}]: TEST Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, test_loss))
            
            if (epoch + 1) % cfg.monitoring.interval == 0:
                # Subset of validation for plotting
                n_plotting = cfg.monitoring.n_plotting
                X_plt = X_v[:n_plotting].cpu().numpy()
                #Y_plt = Y[:n_plotting].cpu().numpy()
                Y_plt_orig = bnn_post.transform_back_mu(Y_v[:n_plotting]).cpu().numpy()
                pred_plt = pred_v[:n_plotting]
                # Slice pred_plt into meaningful Gaussian parameters for this batch
                bnn_post.set_sliced_pred(pred_plt)
                mu_orig = bnn_post.transform_back_mu(bnn_post.mu).cpu().numpy()
                # Log train and val metrics
                loss_dict = {'train': train_loss, 'val': val_loss}
                if cfg.data.test_dir is not None:
                    loss_dict.update(test=test_loss)
                logger.add_scalars('metrics/loss', loss_dict, epoch)
                #rmse = train_utils.get_rmse(mu, Y_plt)
                rmse_dist = train_utils.get_rmse(mu_orig, Y_plt_orig, False)
                rmse_dict = {
                           #'rmse': rmse,
                           'rmse_orig1': np.mean(rmse_dist),
                           'rmse_std': np.std(rmse_dist),
                           'rmse_lens_x': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 0),
                           'rmse_src_x': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 1),
                           'rmse_lens_y': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 2),
                           'rmse_src_y': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 3),
                           'rmse_gamma': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 4),
                           'rmse_e1': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 6),
                           'rmse_e2': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 7),
                           'rmse_psi1': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 8),
                           'rmse_psi2': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 9),
                           }
                # Log second Gaussian stats
                if cfg.model.likelihood_class in ['DoubleGaussianNLL', 'DoubleLowRankGaussianNLL']:
                    logger.add_histogram('val_pred/weight_gaussian2', bnn_post.w2.cpu().numpy(), epoch)
                    mu2_orig = bnn_post.transform_back_mu(bnn_post.mu2).cpu().numpy()
                    rmse_orig2 = train_utils.get_rmse(mu2_orig, Y_plt_orig)
                    rmse_dict.update(rmse_orig2=rmse_orig2)
                logger.add_scalars('metrics/rmse', rmse_dict, epoch)
                # Log log determinant of the covariance matrix
                logdet = train_utils.get_logdet(pred_v[:, Y_dim:].cpu().numpy(), Y_dim)
                logger.add_histogram('logdet_cov_mat', logdet, epoch)
                # Log histograms of named parameters
                if cfg.monitoring.weight_distributions:
                    for param_name, param in net.named_parameters():
                        logger.add_histogram(param_name, param.clone().cpu().data.numpy(), epoch)
                # Log sample images
                if cfg.monitoring.sample_images:
                    sample_img = X_plt[:5]
                    #pred = pred.cpu().numpy()
                    logger.add_images('val_images', sample_img, epoch, dataformats='NCHW')
                # Log 1D marginal mapping
                if cfg.monitoring.marginal_1d_mapping:
                    for param_idx, param_name in enumerate(cfg.data.Y_cols):
                        tag = '1d_mapping/{:s}'.format(param_name)
                        fig = train_utils.get_1d_mapping_fig(param_name, mu_orig[:, param_idx], Y_plt_orig[:, param_idx])
                        logger.add_figure(tag, fig, global_step=epoch)

            if (epoch + 1) % cfg.checkpoint.interval == 0:
                # FIXME compare to last saved epoch val loss
                if val_loss < last_saved_val_loss:
                    os.remove(model_path) if os.path.exists(model_path) else None
                    model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, cfg.model.architecture, epoch)
                    last_saved_val_loss = val_loss
        '''

        # Step lr_scheduler every epoch
        lr_scheduler.step(train_loss)

    #logger.close()
    # Save final state dict
    #if val_loss < last_saved_val_loss:
    #    os.remove(model_path) if os.path.exists(model_path) else None
    #    model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, cfg.model.architecture, epoch)
    #    print("Saved model at {:s}".format(os.path.abspath(model_path)))


if __name__ == '__main__':
    main()
