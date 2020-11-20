import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
#from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Dataset, Data, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

cosmodc2_dir = '/home/jwp/stage/sl/extranet/data/cosmodc2'

class CosmoDC2Sightline(Data):
    """Graph representing a single sightline

    """ 
    def __init__(self, x, edge_index, y):
        """

        Parameters
        ----------
        x : `torch.FloatTensor` of shape `[n_nodes, n_features]`
            the galaxy defining the sightline (first node = v0) and its neighbors
        edge_index : `torch.LongTensor` of shape `[2, n_edges]`
            directed edges from each of the neighbors to v0
        y : `torch.FloatTensor` of shape `[1]`
            the label to infer

        """
        super(CosmoDC2Sightline, self).__init__(x=x, edge_index=edge_index, y=y)

class CosmoDC2(Dataset):
    """Set of graphs representing a subset or all of the CosmoDC2 field

    """

    # https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/catalog_configs/cosmoDC2_v1.1.4_small.yaml
    #healpix_available = [9559,  9686,  9687,  9814,  9815,  9816,  9942,  9943, 10070, 10071, 10072, 10198, 10199, 10200, 10326, 10327, 10450] # small_v1.14
    # https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/catalog_configs/cosmoDC2_v1.1.4_image.yaml
    healpix_available = [8786, 8787, 8788, 8789, 8790, 8791, 8792, 8793, 8794, 8913, 8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 9042, 9043, 9044, 9045, 9046, 9047, 9048, 9049, 9050, 9169, 9170, 9171, 9172, 9173, 9174, 9175, 9176, 9177, 9178, 9298, 9299, 9300, 9301, 9302, 9303, 9304, 9305, 9306, 9425, 9426, 9427, 9428, 9429, 9430, 9431, 9432, 9433, 9434, 9554, 9555, 9556, 9557, 9558, 9559, 9560, 9561, 9562, 9681, 9682, 9683, 9684, 9685, 9686, 9687, 9688, 9689, 9690, 9810, 9811, 9812, 9813, 9814, 9815, 9816, 9817, 9818, 9937, 9938, 9939, 9940, 9941, 9942, 9943, 9944, 9945, 9946, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073, 10074, 10193, 10194, 10195, 10196, 10197, 10198, 10199, 10200, 10201, 10202, 10321, 10322, 10323, 10324, 10325, 10326, 10327, 10328, 10329, 10444, 10445, 10446, 10447, 10448, 10449, 10450, 10451, 10452] # full, 440 sq. deg. cosmoDC2

    columns = ['ra', 'dec', 'bulge_to_total_ratio_i']
    #columns += ['ellipticity_1_bulge_true', 'ellipticity_1_disk_true', 
    #            'ellipticity_2_bulge_true', 'ellipticity_2_disk_true',
    #            'ellipticity_1_true', 'ellipticity_2_true',]
    #columns += ['size_bulge_true', 'size_disk_true', 'size_minor_bulge_true', 'size_minor_disk_true', 'size_minor_true', 'size_true']
    columns += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']

    def __init__(self, root, healpix_pixels, aperture_size, n_data, random_seed, transform=None, pre_transform=None, pre_filter=None):
        self.healpix_pixels = healpix_pixels
        self.aperture_size = aperture_size # arcmin
        self.v0_hp_sampling_radius = 0.5*1.8323*60.0 - (self.aperture_size + 1.0) # radius of hp - (aperture radius + buffer)
        self.n_v0_hp = n_data
        self.random_seed = random_seed
        super(CosmoDC2, self).__init__(root, transform, pre_transform, pre_filter)
        if set(healpix_pixels) - set(self.healpix_available):
            raise ValueError("At least one of the queried healpix pixels is not available.")
        
    @property
    def raw_file_names(self):
        """A list of files relative to self.raw_dir which needs to be found in order to skip the download

        """
        return ['cosmodc2_trainval_{:d}.csv'.format(h) for h in self.healpix_pixels]

    @property
    def processed_file_names(self):
        return ['data_{:d}.pt'.format(n) for n in range(self.n_v0_hp*len(self.healpix_pixels))]

    def download(self):
        # Download to `self.raw_dir`.
        raise ValueError("CosmoDC2 CSV file(s) not found.")

    def process(self):
        i = 0
        for raw_path in self.raw_paths: # iterates over healpy pixels
            # Read in cosmodc2 CSV file at raw_path
            cosmodc2_hp = pd.read_csv(raw_path, index_col=None, usecols=self.columns, nrows=100000)
            # Sample self.n_v0_hp number of v0 galaxies from middle of hp
            cosmodc2_hp.loc[:, ['ra', 'dec']] = cosmodc2_hp[['ra', 'dec']]*60.0 # deg to arcmin
            mag_i_hp = cosmodc2_hp['mag_i_lsst'].values
            center_hp = cosmodc2_hp[['ra', 'dec']].median().values.reshape([1, -1]) # center of hp, shape [1, 2]
            distance_to_center_hp = np.linalg.norm(cosmodc2_hp[['ra', 'dec']].values - center_hp, axis=1) # [N,]
            print("Choosing {:d} LOS galaxies from {:d} within buffered center...".format(self.n_v0_hp, cosmodc2_hp[distance_to_center_hp < self.v0_hp_sampling_radius].shape[0]))
            v0_gals_hp = cosmodc2_hp[distance_to_center_hp < self.v0_hp_sampling_radius].sample(self.n_v0_hp, random_state=self.random_seed).reset_index(drop=True) # [self.n_v0_hp]
            print(v0_gals_hp.shape)
            for v0_hp_idx in range(self.n_v0_hp):
                # Define neighbors in aperture around v0_hp
                v0_hp_pos = v0_gals_hp.loc[v0_hp_idx, ['ra', 'dec']].values.reshape([1, -1]) # [1, 2]
                distances_to_v0_hp = np.linalg.norm(cosmodc2_hp[['ra', 'dec']].values - v0_hp_pos, axis=1) # [N,]
                keep_dist = np.logical_and(distances_to_v0_hp < self.aperture_size, distances_to_v0_hp > 0.05)
                keep_i_mag = np.logical_and(mag_i_hp < 23.0, mag_i_hp > 18.0)
                keep = np.logical_and(keep_dist, keep_i_mag)
                neighbors = cosmodc2_hp.loc[keep, self.columns].reset_index(drop=True) # [n_neighbors,]
                n_neighbors = neighbors.shape[0]
                dist = distances_to_v0_hp[keep] # [n_neighbors,]
                neighbors.loc[:, ['ra', 'dec']] -= v0_hp_pos # define neighbor coordinates as offset from sightline
                v0_gals_hp.loc[:, ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']] -= 20.0
                v0_gals_hp.loc[:, ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']] /= 5.0
                neighbors.loc[:, ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']] -= 20.0
                neighbors.loc[:, ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']] /= 5.0
                x = np.concatenate([v0_gals_hp.loc[v0_hp_idx, self.columns].values.reshape([1, -1]), neighbors.values], axis=0)
                y = np.sum(1.0/np.ones_like(dist), keepdims=True) # artificial summary statistics label
                edge_index = np.concatenate([np.arange(1, n_neighbors+1).reshape([1, -1]), np.zeros([1, n_neighbors], dtype=np.int8)], axis=0) # [2, n_edges=n_neighbors]
                # Convert into Tensor
                x = torch.from_numpy(x).to(torch.float)
                y = torch.from_numpy(y).to(torch.float)
                edge_index = torch.from_numpy(edge_index).to(torch.long)
                data = CosmoDC2Sightline(x, edge_index, y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

train_dataset = CosmoDC2(
                         root=cosmodc2_dir,
                         healpix_pixels=[10450], 
                         aperture_size=3.0, # raytracing res (0.85') x FoV factor (15) = 12.75'
                         n_data=1000,
                         random_seed=1234
                         )  

val_dataset = CosmoDC2(
                         root=cosmodc2_dir,
                         healpix_pixels=[10450], #[10450], 
                         aperture_size=3.0, # raytracing res (0.85') x FoV factor (15) = 12.75'
                         n_data=1000,
                         random_seed=5678
                         )  
print("Number of features: ", val_dataset.num_features)
print("Number of training examples: ", train_dataset.n_v0_hp)
print("Number of validation examples: ", val_dataset.n_v0_hp)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, dropout=0.0):
        super(SAGE, self).__init__()

        self.num_layers = 1 # k=1
        self.batch_size = batch_size
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, 16, normalize=False, aggr='add'))
        self.fc1 = nn.Linear(16, 16, bias=True)
        self.fc2 = nn.Linear(16, 16, bias=True)
        self.fc3 = nn.Linear(16, out_channels, bias=True)
        
    def forward(self, x, edge_index):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        x_target = x[:self.batch_size]  # Target nodes are always placed first.
        x = self.convs[0]((x, x_target), edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        return x

logger = SummaryWriter()
device = torch.device('cuda')
model = SAGE(train_dataset.num_features, out_channels=2, batch_size=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5.e-4, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10, cooldown=10, min_lr=1e-7, verbose=True)

if False:
    for b in train_loader:
        print(b['x'].shape, b['edge_index'].shape)
        model(b['x'].to(device), b['edge_index'].to(device))
        sys.exit()


@torch.no_grad()
def test():
    model.eval()
    val_loss = 0.0
    n_iter = 0
    #mc_dropout = 20
    val_pred = [] #np.empty([mc_dropout*len(val_loader)])
    val_label = [] #np.empty([len(val_loader)])
    val_sigma = [] #np.empty([mc_dropout*len(val_loader)])
    #for d in range(20):
    for batch_i, batch in enumerate(val_loader):
        n_iter += 1
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad()
        out = model(x, edge_index)
        val_pred.append(float(out[0, 0].squeeze()))
        val_label.append(float(y.squeeze()))
        val_sigma.append( float( torch.exp(out[0, 1].squeeze())**0.5 ) )
        loss = nll_diagonal(y.squeeze(), out[0, 0], out[0, 1]) #F.mse_loss(out.squeeze(0), y)
        val_loss += (loss.detach().item() - val_loss)/(1 + n_iter)
    return val_loss, np.array(val_pred), np.array(val_label), np.array(val_sigma)

def nll_diagonal(target, mu, logvar):
    """Evaluate the NLL for single Gaussian with diagonal covariance matrix
    Parameters
    ----------
    target : torch.Tensor of shape [batch_size, Y_dim]
        Y labels
    mu : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the mu (mean parameter) of the BNN posterior
    logvar : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the log of the diagonal elements of the covariance matrix
    Returns
    -------
    torch.Tensor of shape
        NLL values
    """
    precision = torch.exp(-logvar)
    # Loss kernel
    loss = precision * (target - mu)**2.0 + logvar
    # Restore prefactors
    loss += np.log(2.0*np.pi)
    loss *= 0.5
    return loss # torch.mean(torch.sum(loss, dim=1), dim=0)

def get_mapping_fig(pred, label):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(label, pred, alpha=0.2)
    ax.axhline(np.mean(label), linestyle='--', color='tab:red')
    ax.plot(label, label, linestyle='--', color='k')
    return fig

n_epochs = 1000
for epoch in tqdm(range(n_epochs)):
    n_iter = 0
    model.train()
    train_loss = 0.0
    train_pred = [] #np.empty([mc_dropout*len(val_loader)])
    train_label = [] #np.empty([len(val_loader)])
    train_sigma = [] #np.empty([mc_dropout*len(val_loader)])
    for batch_i, batch in enumerate(train_loader):
        n_iter += 1
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = nll_diagonal(y.squeeze(), out[0, 0], out[0, 1]) #F.mse_loss(out.squeeze(0), y)
        train_pred.append(float(out[0, 0].squeeze()))
        train_label.append(float(y.squeeze()))
        train_sigma.append( float( torch.exp(out[0, 1].squeeze())**0.5 ) )
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().item() - train_loss)/(1 + n_iter)
        if n_iter % 500 == 0:
            tqdm.write("Iter [{}/{}/{}]: TRAIN Loss: {:.4f}".format(n_iter, epoch+1, n_epochs, train_loss))
    val_loss, val_pred, val_label, val_sigma = test()
    train_pred, train_label, train_sigma = np.array(train_pred), np.array(train_label), np.array(train_sigma)
    logger.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
    logger.add_scalars('val output', {'mean pred': np.mean(val_pred), 'std pred': np.std(val_pred),
                       'mean uncertainty': np.mean(val_sigma), 'std uncertainty': np.std(val_sigma)}, epoch)
    logger.add_histogram('val_error', val_pred - val_label, epoch)
    logger.add_histogram('val_z', (val_pred - val_label)/val_sigma, epoch)
    #random_idx = np.random.randint(0, len(train_pred)-1, size=1000)
    logger.add_figure('val_mapping', get_mapping_fig(val_pred, val_label), global_step=epoch, close=True)
    logger.add_figure('train_mapping', get_mapping_fig(train_pred, train_label), global_step=epoch, close=True)
    lr_scheduler.step(train_loss)
    if (epoch)%20 == 0:
        tqdm.write(f'Epoch {epoch:02d}/10000, TRAIN Loss: {train_loss:.4f}')
        tqdm.write(f'Epoch {epoch:02d}/10000, VALID Loss: {val_loss:.4f}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, 'trained_model.pt')
logger.close()
