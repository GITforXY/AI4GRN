import torch
from torch_geometric.data import Dataset, InMemoryDataset
from utils import *
import pandas as pd

class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = '{}_data'.format(self.split)
        else:
            name = '{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=self.data.x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data


def LoadDatasets(
    dynamic,
    path,
    data,
    split_edge,
    num_hops,
    percent,
    node_label,
    ratio_per_hop,
    max_nodes_per_hop,
    directed,
    split=['train', 'valid', 'test']
):
    datasets = []
    for data_type in split:
        if dynamic:
            datasets.append(SEALDynamicDataset(path, data, split_edge, num_hops,
                                               percent, data_type, node_label, ratio_per_hop,
                                               max_nodes_per_hop, directed))
        else:
            datasets.append(SEALDataset(path, data, split_edge, num_hops,
                                        percent, data_type, node_label, ratio_per_hop,
                                        max_nodes_per_hop, directed))

    return datasets



def load_data(feat_path, edge_paths):
    feat = pd.read_csv(feat_path, index_col=0)
    x = torch.tensor(feat.values, dtype=torch.float)

    split_edge = load_edge_split(edge_paths)

    edges = split_edge['train']['edge'].clone().detach()
    edge_index = edges.t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data, split_edge


def load_edge_split(edge_paths):
    train_df = pd.read_csv(edge_paths['train'], index_col=0)
    val_df = pd.read_csv(edge_paths['valid'], index_col=0)
    test_df = pd.read_csv(edge_paths['test'], index_col=0)

    train_edges = torch.tensor(train_df.values[:, :-1], dtype=torch.long)
    val_edges = torch.tensor(val_df.values[:, :-1], dtype=torch.long)
    test_edges = torch.tensor(test_df.values[:, :-1], dtype=torch.long)
    train_labels = train_df.values[:, -1]
    val_labels = val_df.values[:, -1]
    test_labels = test_df.values[:, -1]

    train_pos = train_edges[train_labels == 1]
    train_neg = train_edges[train_labels == 0]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = torch.concatenate([train_pos, train_pos[:, [-1, 0]]])
    split_edge['train']['edge_neg'] = torch.concatenate([train_neg, train_neg[:, [-1, 0]]])
    split_edge['valid']['edge'] = val_edges[val_labels == 1]
    split_edge['valid']['edge_neg'] = val_edges[val_labels == 0]
    split_edge['test']['edge'] = test_edges[test_labels == 1]
    split_edge['test']['edge_neg'] = test_edges[test_labels == 0]
    return split_edge

