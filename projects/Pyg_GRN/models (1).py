# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,
                      Sequential, BatchNorm1d as BN)
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv, GATv2Conv, GraphConv,
                                global_sort_pool, global_add_pool, global_mean_pool)


class NGNN_GCNConv(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NGNN_GCNConv, self).__init__()
        self.conv = GCNConv(input_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, output_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x, edge_weight=None):
        x = self.conv(g, x, edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(GCN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, num_features, use_ignn=False, use_gatv2=False,
                 k=10, GNN=GCNConv, NGNN=NGNN_GCNConv, GATv2=GATv2Conv, use_feature=False, node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.num_heads = 4


        self.convs = ModuleList()
        if use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1
        elif use_gatv2:
            self.convs.append(GATv2(initial_channels, hidden_channels, heads=self.num_heads))
            for i in range(0, num_layers - 1):
                self.convs.append(GATv2(hidden_channels * self.num_heads, hidden_channels, heads=self.num_heads))
            self.convs.append(GATv2(hidden_channels * self.num_heads, 1, heads=self.num_heads))
            total_latent_dim = (hidden_channels * num_layers + 1) * self.num_heads
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1

        conv1d_channels = [16, 32]
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)

        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN_feat(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, num_features, use_ignn=False, use_gatv2=False,
                 k=10, GNN=GCNConv, NGNN=NGNN_GCNConv, GATv2=GATv2Conv, node_embedding=None):
        super(DGCNN_feat, self).__init__()

        self.node_embedding = node_embedding
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        # self.feat = torch.nn.Sequential(Linear(num_features, 128), ReLU(), Linear(128, hidden_channels))
        self.feat = torch.nn.Sequential(Linear(num_features, hidden_channels), ReLU())
        initial_channels = 2*hidden_channels
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.num_heads = 4

        self.convs = ModuleList()
        if use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1
        elif use_gatv2:
            self.convs.append(GATv2(initial_channels, hidden_channels, heads=self.num_heads))
            for i in range(0, num_layers - 1):
                self.convs.append(GATv2(hidden_channels * self.num_heads, hidden_channels, heads=self.num_heads))
            self.convs.append(GATv2(hidden_channels * self.num_heads, 1, heads=self.num_heads))
            total_latent_dim = (hidden_channels * num_layers + 1) * self.num_heads
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1

        conv1d_channels = [16, 32]
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        x = self.feat(x)
        x = torch.cat([z_emb, x.to(torch.float)], 1)

        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]  # 残差连接？
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hiddendim]
        x = F.relu(self.conv1(x))  # [num_graphs, 16, k]
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class DGCNN_feat_noNeigFeat(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, num_features, use_ignn=False, use_gatv2=False,
                 k=10, GNN=GCNConv, NGNN=NGNN_GCNConv, GATv2=GATv2Conv, node_embedding=None):
        super(DGCNN_feat_noNeigFeat, self).__init__()

        self.node_embedding = node_embedding
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        # self.feat = torch.nn.Sequential(Linear(num_features, 128), ReLU(), Linear(128, hidden_channels))
        self.lin01 = Linear(num_features, hidden_channels)  # FFN单独编码feature
        self.lin02 = Linear(num_layers * hidden_channels + 1 + hidden_channels,
                            num_layers * hidden_channels + 1)  # 3*32+1+32, 3*32+1

        self.feat = torch.nn.Sequential(Linear(num_features, hidden_channels), ReLU())
        initial_channels = hidden_channels
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.num_heads = 4

        self.convs = ModuleList()
        if use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1
        elif use_gatv2:
            self.convs.append(GATv2(initial_channels, hidden_channels, heads=self.num_heads))
            for i in range(0, num_layers - 1):
                self.convs.append(GATv2(hidden_channels * self.num_heads, hidden_channels, heads=self.num_heads))
            self.convs.append(GATv2(hidden_channels * self.num_heads, 1, heads=self.num_heads))
            total_latent_dim = (hidden_channels * num_layers + 1) * self.num_heads
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1



        conv1d_channels = [16, 32]
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x, edge_weight=None, node_id=None):
        x_feat = F.relu(self.lin01(x))
        x = self.z_embedding(z)

        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # linear -> concat -> linear
        x = torch.cat([x, x_feat], dim=1)
        x = F.relu(self.lin02(x))

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class DGCNN_feat_rec(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, num_features, use_ignn=False, use_gatv2=False,
                 k=10, GNN=GCNConv, NGNN=NGNN_GCNConv, GATv2=GATv2Conv, node_embedding=None):
        super(DGCNN_feat_rec, self).__init__()

        self.node_embedding = node_embedding
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.feat = torch.nn.Sequential(Linear(num_features, 128), ReLU(), Linear(128, hidden_channels))
        # self.feat = torch.nn.Sequential(Linear(num_features, hidden_channels), ReLU())
        self.feat_rec = torch.nn.Sequential(Linear(hidden_channels, 128), ReLU(), Linear(128, num_features), ReLU())

        initial_channels = 2*hidden_channels
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.num_heads = 4

        self.convs = ModuleList()
        if use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1
        elif use_gatv2:
            self.convs.append(GATv2(initial_channels, hidden_channels, heads=self.num_heads))
            for i in range(0, num_layers - 1):
                self.convs.append(GATv2(hidden_channels * self.num_heads, hidden_channels, heads=self.num_heads))
            self.convs.append(GATv2(hidden_channels * self.num_heads, 1, heads=self.num_heads))
            total_latent_dim = (hidden_channels * num_layers + 1) * self.num_heads
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))
            total_latent_dim = hidden_channels * num_layers + 1

        conv1d_channels = [16, 32]
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        x = self.feat(x)
        x_rec = self.feat_rec(x)
        x = torch.cat([z_emb, x.to(torch.float)], 1)

        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x, x_rec

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, train_eps=False):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        if self.jk:
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = global_mean_pool(xs[-1], batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


