#!/usr/bin/env python3
# Modified by @jiwoncpark from an example `higher` script
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

import time
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from extranet.trainval_data.cosmodc2_data import CosmoDC2
from extranet.models.gcn import GCN
from extranet.losses.nll import nll_diagonal
import higher
from torch.utils.data.sampler import SubsetRandomSampler


def main():
    # n_way, k_spt, k_qry, task_num, seed

    #seed = 1234
    #torch.manual_seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    device = torch.device('cuda')

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

    train_support_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=True, sampler=SubsetRandomSampler(np.arange(800).tolist()))
    train_query_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=True, sampler=SubsetRandomSampler(np.arange(800, 1000).tolist()))
    val_support_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, drop_last=True, sampler=SubsetRandomSampler(np.arange(800).tolist()))
    val_query_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, drop_last=True, sampler=SubsetRandomSampler(np.arange(800, 1000).tolist()))

    if False:
        for b in train_support_loader:
            print(b['x'].shape, b['edge_index'].shape)
            #print("Nodes: ", b['x'])
            print("Edges: ", b['edge_index'])
            #model(b['x'].to(device), b['edge_index'].to(device))
            sys.exit()

    logger = SummaryWriter()
    model = GCN(train_dataset.num_features, out_channels=2, batch_size=1, dropout=0.0)
    model = model.to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=5.e-4, weight_decay=1e-5)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, mode='min', factor=0.5, patience=10, cooldown=10, min_lr=1e-7, verbose=True)
    n_epochs = 10
    for epoch in tqdm(range(n_epochs)):
        train_log = train(train_support_loader, train_query_loader, model, device, meta_optimizer, epoch)
        val_log = test(val_support_loader, val_query_loader, model, device, epoch)
        if (epoch)%2 == 0:
            tqdm.write('Epoch {:02d}/10000, TRAIN Loss: {:.4f}'.format(epoch, train_log['query_loss']))
            tqdm.write('Epoch {:02d}/10000, VALID Loss: {:.4f}'.format(epoch, val_log['query_loss']))
        logger.add_scalars('loss', {'train': train_log['query_loss'], 'val': val_log['query_loss']}, epoch)
        #lr_scheduler.step(train_log['loss'])

def train(support_loader, query_loader, model, device, meta_opt, epoch):
    model.train()
    n_iter = 0
    total_query_loss = 0.0
    for batch_idx, (batch_s, batch_q) in enumerate(zip(support_loader, query_loader)):
        # Sample a batch of support and query images and labels.
        x_s = batch_s['x'].to(device)
        edge_index_s = batch_s['edge_index'].to(device)
        y_s = batch_s['y'].to(device)
        x_q = batch_q['x'].to(device)
        edge_index_q = batch_q['edge_index'].to(device)
        y_q = batch_q['y'].to(device)

        task_num = y_s.shape[0]

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(model.parameters(), lr=1e-5)

        losses_q = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    out_s = fnet(x_s, edge_index_s)
                    loss_s = nll_diagonal(y_s, out_s[:, 0], out_s[:, 1])
                    diffopt.step(loss_s)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                out_q = fnet(x_q, edge_index_q)
                loss_q = nll_diagonal(y_q, out_q[:, 0], out_q[:, 1])
                losses_q.append(loss_q.detach())

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                loss_q.backward()
                n_iter += 1
                total_query_loss += (loss_q.detach().item() - total_query_loss)/(1 + n_iter)

        meta_opt.step()

    log = {
        'query_loss': total_query_loss,
    }
    return log


def test(support_loader, query_loader, model, device, epoch):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    model.train()
    n_iter = 0
    total_query_loss = 0.0
    for batch_idx, (batch_s, batch_q) in enumerate(zip(support_loader, query_loader)):
        x_s = batch_s['x'].to(device)
        edge_index_s = batch_s['edge_index'].to(device)
        y_s = batch_s['y'].to(device)
        x_q = batch_q['x'].to(device)
        edge_index_q = batch_q['edge_index'].to(device)
        y_q = batch_q['y'].to(device)

        task_num = y_s.shape[0]

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(model.parameters(), lr=1e-1)

        losses_q = []
        for i in range(task_num):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    out_s = fnet(x_s, edge_index_s)
                    loss_s = nll_diagonal(y_s, out_s[:, 0], out_s[:, 1])
                    diffopt.step(loss_s)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                out_q = fnet(x_q, edge_index_q)
                loss_q = nll_diagonal(y_q, out_q[:, 0], out_q[:, 1])
                losses_q.append(loss_q.detach())
                n_iter += 1
                total_query_loss += (loss_q.detach().item() - total_query_loss)/(1 + n_iter)

    log = {
        'query_loss': total_query_loss,
    }
    return log

# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()
