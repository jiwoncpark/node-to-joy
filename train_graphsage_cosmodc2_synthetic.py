import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
#from torch_geometric.data import NeighborSampler
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from extranet.trainval_data.cosmodc2_data import CosmoDC2
from extranet.models.gcn import GCN
from extranet.losses.nll import nll_diagonal

train_dataset = CosmoDC2(
                         root='/home/jwp/stage/sl/extranet/data/cosmodc2_train',
                         healpix_pixels=[10450], 
                         aperture_size=3.0, # raytracing res (0.85') x FoV factor (15) = 12.75'
                         n_data=1000,
                         random_seed=1234
                         )  
val_dataset = CosmoDC2(
                         root='/home/jwp/stage/sl/extranet/data/cosmodc2_val',
                         healpix_pixels=[10327],
                         aperture_size=3.0, # raytracing res (0.85') x FoV factor (15) = 12.75'
                         n_data=1000,
                         random_seed=5678
                         )  
#print("Number of features: ", val_dataset.num_features)
print("Number of training examples: ", train_dataset.n_v0_hp)
print("Number of validation examples: ", val_dataset.n_v0_hp)
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

logger = SummaryWriter()
device = torch.device('cuda')
model = GCN(train_dataset.num_features, out_channels=2, batch_size=batch_size, dropout=0.0)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5.e-4, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, cooldown=10, min_lr=1e-7, verbose=True)

if False:
    b = train_loader.next()
    print(b['x'].shape, "works?")
    for b in train_loader:
        print(b['x'].shape, b['edge_index'].shape)
        #print("Nodes: ", b['x'])
        print("Edges: ", b['edge_index'])
        #model(b['x'].to(device), b['edge_index'].to(device))
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
        val_pred += [out[:, 0].detach().cpu().numpy().squeeze()]
        val_label += [y.detach().cpu().numpy().squeeze()]
        val_sigma += [(torch.exp(out[:, 1])**0.5).detach().cpu().numpy().squeeze()]
        loss = nll_diagonal(y, out[:, 0], out[:, 1]) #F.mse_loss(out.squeeze(0), y)
        val_loss += (loss.detach().item() - val_loss)/(1 + n_iter)
    return val_loss, np.array(val_pred), np.array(val_label), np.array(val_sigma)

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
        loss = nll_diagonal(y, out[:, 0], out[:, 1]) #F.mse_loss(out.squeeze(0), y)
        train_pred += [out[:, 0].detach().cpu().numpy().squeeze()]
        train_label += [y.detach().cpu().numpy().squeeze()]
        train_sigma += [(torch.exp(out[:, 1])**0.5).detach().cpu().numpy().squeeze()]
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
    logger.add_figure('train_mapping', get_mapping_fig(train_pred[:1000], train_label[:1000]), global_step=epoch, close=True)
    lr_scheduler.step(train_loss)
    if (epoch)%20 == 0:
        tqdm.write(f'Epoch {epoch:02d}/10000, TRAIN Loss: {train_loss:.4f}')
        tqdm.write(f'Epoch {epoch:02d}/10000, VALID Loss: {val_loss:.4f}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, 'trained_model_test.pt')
logger.close()
