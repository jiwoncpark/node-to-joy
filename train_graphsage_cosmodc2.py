import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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

    columns = ['ra', 'dec', 'bulge_to_total_ratio_i', 'stellar_mass', 'stellar_mass_bulge', 'stellar_mass_disk',]
    columns += ['ellipticity_1_bulge_true', 'ellipticity_1_disk_true', 
                'ellipticity_2_bulge_true', 'ellipticity_2_disk_true',
                'ellipticity_1_true', 'ellipticity_2_true',]
    columns += ['size_bulge_true', 'size_disk_true', 'size_minor_bulge_true', 'size_minor_disk_true', 'size_minor_true', 'size_true']
    columns += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']

    def __init__(self, root, healpix_pixels, aperture_size, transform=None, pre_transform=None, pre_filter=None):
        self.healpix_pixels = healpix_pixels
        self.aperture_size = aperture_size # arcmin
        self.v0_hp_sampling_radius = 0.5*1.8323*60.0 - (self.aperture_size + 1.0) # radius of hp - (aperture + buffer)
        self.n_v0_hp = 1000
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
            cosmodc2_hp = pd.read_csv(raw_path, index_col=None, usecols=self.columns, nrows=50000)
            # Sample self.n_v0_hp number of v0 galaxies from middle of hp
            cosmodc2_hp.loc[:, ['ra', 'dec']] = cosmodc2_hp[['ra', 'dec']]*60.0 # deg to arcmin
            mag_i_hp = cosmodc2_hp['mag_i_lsst'].values
            center_hp = cosmodc2_hp[['ra', 'dec']].median().values.reshape([1, -1]) # center of hp, shape [1, 2]
            distance_to_center_hp = np.linalg.norm(cosmodc2_hp[['ra', 'dec']].values - center_hp, axis=1) # [N,]
            v0_gals_hp = cosmodc2_hp[distance_to_center_hp < self.aperture_size].sample(self.n_v0_hp).reset_index(drop=True) # [self.n_v0_hp]
            print(v0_gals_hp.shape)
            for v0_hp_idx in range(self.n_v0_hp):
                # Define neighbors in aperture around v0_hp
                v0_hp_pos = v0_gals_hp.loc[v0_hp_idx, ['ra', 'dec']].values.reshape([1, -1]) # [1, 2]
                distances_to_v0_hp = np.linalg.norm(cosmodc2_hp[['ra', 'dec']].values - v0_hp_pos, axis=1) # [N,]
                keep_dist = np.logical_and(distances_to_v0_hp < self.aperture_size, distances_to_v0_hp > 0.05)
                keep_r = True #np.logical_and(mag_i_hp < 23.0, mag_i_hp > 18.0)
                keep = np.logical_and(keep_dist, keep_r)
                neighbors = cosmodc2_hp.loc[keep, self.columns] # [n_neighbors,]
                n_neighbors = neighbors.shape[0]
                dist = distances_to_v0_hp[keep] # [n_neighbors,]
                x = np.concatenate([v0_gals_hp.loc[v0_hp_idx, self.columns].values.reshape([1, -1]), neighbors.values], axis=0)
                y = np.sum(1.0/dist, keepdims=True) # artificial summary statistics label
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
            plt.show()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

train_dataset = CosmoDC2(
                         root=cosmodc2_dir,
                         healpix_pixels=[9559], 
                         aperture_size=12.75, # raytracing res (0.85') x FoV factor (15)
                         )  

val_dataset = CosmoDC2(
                         root=cosmodc2_dir,
                         healpix_pixels=[9559], #[10450], 
                         aperture_size=12.75, # raytracing res (0.85') x FoV factor (15)
                         )  
print("Number of features: ", val_dataset.num_features)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for layer_i in range(self.num_layers):
            x = self.convs[layer_i](x, edge_index)
            if layer_i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

logger = SummaryWriter()
device = torch.device('cuda')
model = SAGE(train_dataset.num_features, hidden_channels=64, out_channels=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=50, min_lr=1e-7, verbose=True)

if False:
    for b in train_loader:
        print(b['x'].shape, b['edge_index'].shape)
        model(b['x'].to(device), b['edge_index'].to(device))
        sys.exit()

def train():
    model.train()

    total_loss = 0.0
    for train_i, batch in enumerate(train_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.mse_loss(torch.mean(out, dim=0), y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)

    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test():
    model.eval()
    total_loss = 0.0
    for val_i, batch in enumerate(val_loader):
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        y = batch['y'].to(device)
        out = model(x, edge_index)
        loss = F.mse_loss(torch.mean(out, dim=0), y)
        total_loss += float(loss)
    loss = total_loss / len(val_loader)
    return loss

for epoch in tqdm(range(10000)):
    train_loss = train()
    val_loss = test()
    logger.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
    lr_scheduler.step(train_loss)
    if (epoch)%100 == 0:
        tqdm.write(f'Epoch {epoch:02d}/10000, TRAIN Loss: {train_loss:.4f}')
        tqdm.write(f'Epoch {epoch:02d}/10000, VALID Loss: {val_loss:.4f}')
logger.close()
