"""Various GNN models

"""
import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GravNetConv
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['GCNNet', 'GATNet', 'SageNet', 'GravNet']


def get_zero_nodes(batch_idx):
    """Get indices of the zeroth nodes in the batch

    """
    batch_idx = torch.cat([torch.zeros(1, device=batch_idx.device), batch_idx])
    diff = batch_idx[1:] - batch_idx[:-1]
    diff[0] = 1
    return diff.bool()


class GCNNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0,
                 kwargs={}):
        super(GCNNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.kwargs = kwargs
        self.convs = nn.ModuleList()
        for i in range(self.n_layers-1):
            n_in = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(GCNConv(n_in,
                                      self.hidden_channels,
                                      aggr='add',
                                      add_self_loops=False,
                                      **self.kwargs))
        # Last layer
        self.convs.append(GCNConv(self.hidden_channels,
                                  self.out_channels,
                                  aggr='add',
                                  add_self_loops=False,
                                  **self.kwargs))
        # self.fc = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.n_layers-1):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.convs[-1](x, edge_index)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        return x


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256,
                 kwargs={}, n_layers=3, dropout=0.0):
        super(GATNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.kwargs = kwargs
        self.convs = nn.ModuleList()
        for i in range(self.n_layers-1):
            n_in = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(GATConv(n_in,
                                      self.hidden_channels,
                                      aggr='add',
                                      dropout=self.dropout,
                                      add_self_loops=False,
                                      **self.kwargs))
        # Last layer
        self.convs.append(GATConv(self.hidden_channels,
                                  self.out_channels,
                                  aggr='add',
                                  dropout=self.dropout,
                                  add_self_loops=False,
                                  **self.kwargs))
        self.alpha = None  # init, [edge_index, attention_weights]
        # self.fc = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.n_layers-1):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x, alpha = self.convs[-1](x, edge_index, return_attention_weights=True)
        self.alpha = alpha
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        return x


class SageNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0,
                 kwargs={}):
        super(SageNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.kwargs = kwargs
        self.convs = nn.ModuleList()
        for i in range(self.n_layers-1):
            n_in = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(SAGEConv(n_in,
                                       self.hidden_channels,
                                       aggr='add',
                                       normalize=True,  # otherwise explode
                                       root_weight=False,
                                       **self.kwargs))
        # Last layer
        self.convs.append(SAGEConv(self.hidden_channels,
                                   self.out_channels,
                                   aggr='add',
                                   normalize=True,  # otherwise explode
                                   root_weight=False,
                                   **self.kwargs))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.n_layers-1):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.convs[-1](x, edge_index)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        return x


class GravNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0,
                 kwargs={}):
        super(GravNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.kwargs = kwargs
        self.convs = nn.ModuleList()
        for i in range(self.n_layers-1):
            n_in = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(GravNetConv(n_in,
                                          self.hidden_channels,
                                          aggr='add',
                                          space_dimensions=3,
                                          propagate_dimensions=2,
                                          k=20,
                                          **self.kwargs))
        # Last layer
        self.convs.append(GravNetConv(self.hidden_channels,
                                      self.out_channels,
                                      aggr='add',
                                      space_dimensions=3,
                                      propagate_dimensions=2,
                                      k=20,
                                      **self.kwargs))

    def forward(self, data):
        x, batch = data.x, data.batch  # edge information not used
        for i in range(self.n_layers-1):
            x = self.convs[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.convs[-1](x)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        return x
