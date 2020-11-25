from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, dropout=0.0):
        super(GCN, self).__init__()

        self.num_layers = 1 # k=1
        self.batch_size = batch_size
        self.dropout = dropout

        self.convs = nn.ModuleList()
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