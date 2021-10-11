from typing import Optional, Tuple
from torch import Tensor
import torch
from torch.nn import (Module, ModuleList, ReLU, LayerNorm,
                      Sequential as Seq, Linear as Lin)
from torch_scatter import scatter_add
from torch_geometric.nn import MetaLayer
from torch import nn
import torch.nn.functional as F
from n2j.models.flow import Flow, MAF, Perm
from n2j.losses.gaussian_nll import DiagonalGaussianNLL


__all__ = ['N2JNet']
DEBUG = False


class MCDropout(nn.Dropout):
    """1D dropout that stays on during training and testing

    """
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class MCDropout2d(nn.Dropout2d):
    """2D dropout that stays on during training and testing

    """
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout2d(input, self.p, True, self.inplace)


class CustomMetaLayer(MetaLayer):
    def __init__(self, node_model=None, global_model=None):
        super(CustomMetaLayer, self).__init__(edge_model=None,
                                              node_model=node_model,
                                              global_model=global_model)
        pass

    def forward(self,
                x: Tensor,
                u: Optional[Tensor] = None,
                batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.node_model is not None:
            x = self.node_model(x, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, u, batch)
        return x, u


class N2JNet(Module):
    def __init__(self, dim_in, dim_out_local, dim_out_global, dim_local, dim_global,
                 dim_hidden=20, dim_pre_aggr=20, n_iter=20, n_out_layers=5,
                 global_flow=False,
                 dropout=0.0,
                 class_weight=None,
                 use_ss=True,
                 dim_in_meta=2):
        """Edgeless graph neural network modeling relationships among nodes
        and between nodes and global

        Parameters
        ----------
        dim_in : int
            number of input features per node
        dim_out_local : int
            number of targets per node
        dim_out_global : int
            number of targets per graph
        dim_local : int
        dim_global : int
        dim_hidden : int
        dim_pre_aggr : int
        n_iter : int
        n_out_layers : int
        global_flow : bool
        dropout : float
            fraction of weights to zero during training and testing,
            for MC dropout. Default: 0.0
        class_weight : torch.tensor
        use_ss : bool, True
            Whether to use summary stats to init global encoding.


        """
        super(N2JNet, self).__init__()
        self.dim_in = dim_in
        self.dim_out_local = dim_out_local
        self.dim_out_global = dim_out_global
        self.dim_hidden = dim_hidden
        self.dim_local = dim_local
        self.dim_global = dim_global
        self.dim_pre_aggr = dim_pre_aggr
        self.n_iter = n_iter
        self.n_out_layers = n_out_layers
        self.global_flow = global_flow
        self.class_weight = class_weight
        self.dropout = dropout
        self.dim_in_meta = dim_in_meta
        # MLP for initially encoding local
        self.mlp_node_init = Seq(Lin(self.dim_in, self.dim_hidden),
                                 ReLU(),
                                 MCDropout(self.dropout),
                                 Lin(self.dim_hidden, self.dim_hidden),
                                 ReLU(),
                                 MCDropout(self.dropout),
                                 Lin(self.dim_hidden, self.dim_local),
                                 LayerNorm(self.dim_local))
        # MLP for initially encoding global
        self.use_ss = use_ss
        if self.use_ss:
            self.mlp_global_init = Seq(Lin(self.dim_in_meta, self.dim_hidden),
                                       ReLU(),
                                       MCDropout(self.dropout),
                                       Lin(self.dim_hidden, self.dim_hidden),
                                       ReLU(),
                                       MCDropout(self.dropout),
                                       Lin(self.dim_hidden, self.dim_global),
                                       LayerNorm(self.dim_global))
        # MLPs for encoding local and global
        meta_layers = ModuleList()
        for i in range(self.n_iter):
            node_model = NodeModel(self.dim_local, self.dim_global,
                                   self.dim_hidden, self.dropout)
            global_model = GlobalModel(self.dim_local, self.dim_global,
                                       self.dim_hidden, self.dim_pre_aggr,
                                       self.dropout)
            meta = CustomMetaLayer(node_model=node_model, global_model=global_model)
            meta_layers.append(meta)
        self.meta_layers = meta_layers
        # Networks for local and global output
        self.net_out_local = Seq(Lin(self.dim_local, self.dim_hidden),
                                 ReLU(),
                                 MCDropout(self.dropout),
                                 Lin(self.dim_hidden, self.dim_hidden),
                                 ReLU(),
                                 MCDropout(self.dropout),
                                 Lin(self.dim_hidden, self.dim_out_local*2))
        if self.global_flow:
            self.net_out_global = Flow(*[[
                                       MAF(self.dim_global, self.dim_out_global, hidden=dim_hidden),
                                       Perm(self.dim_global)][i%2] for i in \
                                       range(self.n_out_layers*2 + 1)])
        else:
            self.net_out_global = Seq(Lin(self.dim_global, self.dim_hidden),
                                      ReLU(),
                                      MCDropout(self.dropout),
                                      Lin(self.dim_hidden, self.dim_hidden),
                                      ReLU(),
                                      MCDropout(self.dropout),
                                      Lin(self.dim_hidden, self.dim_out_global*2))
        # Losses
        self.local_nll = DiagonalGaussianNLL(dim_out_local)
        self.global_nll = DiagonalGaussianNLL(dim_out_global)

    def forward(self, data):
        x = data.x  # [n_nodes, n_features]
        x_meta = data.x_meta  # [batch_size, 2]
        batch = data.batch  # [batch_size,]
        batch_size = data.y.shape[0]
        # Init node and global encodings x, u
        x = self.mlp_node_init(x)  # [n_nodes, dim_local]
        if self.use_ss:
            u = self.mlp_global_init(x_meta)
        else:
            u = torch.zeros(batch_size, self.dim_global).to(x.dtype).to(x.device)
        for i, meta in enumerate(self.meta_layers):
            x, u = meta(x=x, u=u, batch=batch)
        # x : [n_nodes, dim_local]
        # u : [batch_size, dim_global]
        if DEBUG:
            print("x is nan:", torch.any(torch.isnan(x)), x.mean())
            print("u is nan:", torch.any(torch.isnan(u)), u.mean())
        x = self.net_out_local(x)  # [n_nodes, dim_local_out*2]
        if not self.global_flow:
            u = self.net_out_global(u)  # [batch_size, dim_global_out*2]
        return x, u

    def local_loss(self, x, data):
        y_local = data.y_local  # [n_nodes, 2]
        nlogp_local = self.local_nll(x, y_local)  # [n_nodes,]
        return nlogp_local

    def global_loss(self, u, data):
        y = data.y
        if self.global_flow:
            u_out, log_det = self.net_out_global(u, y)
            if DEBUG:
                print("u_out", torch.any(torch.isnan(u_out)), u_out.mean())
                print("log_det", torch.any(torch.isnan(log_det)), log_det.mean())
            log_prob = -u_out.pow(2).sum(1)/2  # Standard normal base dist
            normalized_log_prob = log_prob + log_det
            nlogp_global = - normalized_log_prob  # [batch_size,]
        else:
            nlogp_global = self.global_nll(u, y)  # [batch_size,]
        return nlogp_global

    def loss(self, x, u, data):
        local_loss = self.local_loss(x, data)  # [n_nodes,]
        local_loss = scatter_add(local_loss, data.batch, dim=0)  # [batch_size,]
        global_loss = self.global_loss(u, data)  # [batch_size,]
        # Weight by inverse class number counts
        if self.class_weight is not None:
            y_weight = 1.0/self.class_weight[data.y_class].squeeze()
            local_loss *= y_weight
            global_loss *= y_weight
        return local_loss.mean(), global_loss.mean()


class NodeModel(Module):
    def __init__(self, dim_local, dim_global, dim_hidden, dropout):
        """MLP governing the node representation

        """
        super(NodeModel, self).__init__()
        self.dim_local = dim_local
        self.dim_global = dim_global
        self.dim_hidden = dim_hidden
        self.dim_concat = self.dim_local + self.dim_global
        self.dropout = dropout
        self.mlp = Seq(Lin(self.dim_concat, self.dim_hidden),
                       LayerNorm(self.dim_hidden),
                       ReLU(),
                       MCDropout(self.dropout),
                       Lin(self.dim_hidden, self.dim_hidden),
                       LayerNorm(self.dim_hidden),
                       ReLU(),
                       MCDropout(self.dropout),
                       Lin(self.dim_hidden, self.dim_local),
                       LayerNorm(self.dim_local))

    def forward(self, x, u, batch):
        # x ~ [n_nodes, dim_local]
        # u ~ [batch, dim_global] but u[batch] ~ [n_nodes, dim_global]
        out = torch.cat([x, u[batch]], dim=-1)  # [n_nodes, dim_local + dim_global]
        out = self.mlp(out) + x  # [n_nodes, dim_local]
        return out


class GlobalModel(Module):
    def __init__(self, dim_local, dim_global, dim_hidden, dim_pre_aggr,
                 dropout):
        """MLP governing the global representation

        """
        super(GlobalModel, self).__init__()
        self.dim_local = dim_local
        self.dim_global = dim_global
        self.dim_hidden = dim_hidden
        self.dim_concat = self.dim_local + self.dim_global
        self.dim_pre_aggr = dim_pre_aggr
        self.dropout = dropout

        # MLP prior to aggregating node encodings
        self.mlp_pre_aggr = Seq(Lin(self.dim_concat, self.dim_hidden),
                                ReLU(),
                                MCDropout(self.dropout),
                                Lin(self.dim_hidden, self.dim_hidden),
                                ReLU(),
                                MCDropout(self.dropout),
                                Lin(self.dim_hidden, self.dim_pre_aggr),
                                LayerNorm(self.dim_pre_aggr))
        # MLP after aggregating node encodings
        self.mlp_post_aggr = Seq(Lin(self.dim_pre_aggr+self.dim_global, self.dim_hidden),
                                 ReLU(),
                                 MCDropout(self.dropout),
                                 Lin(self.dim_hidden, self.dim_hidden),
                                 ReLU(),
                                 MCDropout(self.dropout),
                                 Lin(self.dim_hidden, self.dim_global),
                                 LayerNorm(self.dim_global))

    def forward(self, x, u, batch):
        out = torch.cat([x, u[batch]], dim=-1)  # [n_nodes, dim_local + dim_global]
        out = self.mlp_pre_aggr(out)  # [n_nodes, self.dim_pre_aggr]
        out = scatter_add(out, batch, dim=0)  # [batch_size, dim_pre_aggr]
        out = torch.cat([out, u], dim=-1)  # [batch_size, dim_pre_aggr + dim_global]
        out = self.mlp_post_aggr(out)  # [batch_size, dim_global]
        out += u  # [batch_size, dim_global]
        return out


if __name__ == '__main__':
    net = N2JNet(dim_in=4, dim_out_local=2, dim_out_global=1,
                 dim_local=11, dim_global=7,
                 dim_hidden=19, dim_pre_aggr=21, n_iter=5,
                 n_out_layers=7)

    class Batch:
        def __init__(self, x, y_local, y, y_class, batch):
            self.x = x
            self.y_local = y_local
            self.y = y
            self.y_class = y_class
            self.batch = batch

    batch = Batch(x=torch.randn(5, 4),
                  y_local=torch.randn(5, 2),
                  y=torch.randn(3, 1),
                  y_class=torch.tensor([2, 3, 3]).long(),
                  batch=torch.LongTensor([0, 0, 1, 1, 2]))

    x, u = net(batch)
    print(x.shape)
    print(u.shape)
    print("local loss: ", net.local_loss(x, batch).shape)
    print("global loss: ", net.global_loss(u, batch).shape)
    local_loss, global_loss = net.loss(x, u, batch)
    print("loss: ", local_loss.shape, global_loss.shape)
    print(local_loss.cpu().item())
    print((local_loss/2.0 + 0.0).item())


    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of params: {n_params}")

