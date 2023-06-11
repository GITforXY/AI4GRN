import argparse
import os
from data import load_data
import math
import torch
from data import LoadDatasets
from torch_geometric.data import DataLoader
from train import LP_GRN
import time
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

# Data settings
parser = argparse.ArgumentParser(description='GRN')
parser.add_argument('--data', type=str, default='hESC', help='data type')
parser.add_argument('--net', type=str, default='Specific', help='network type')
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--use_pca', action='store_true',
                    help="whether to use PCA node features as GNN input")
parser.add_argument('--hidden_channels', type=int, default=32)

# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl',
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true',
                    help="whether to consider edge weight in GNN")

# Training settings
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--dynamic', action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")


args = parser.parse_args()

if args.use_pca:
    args.use_feature = True
# Load data
data_type, net_type, num = args.data, args.net, str(args.num)
data_name = net_type + '_' + data_type + '_' + num

feat_path = '/mnt/data/oss_beijing/qiank/Dataset/Benchmark Dataset/' + \
            net_type + ' Dataset/' + data_type + '/TFs+' + num + '/BL--ExpressionData.csv'
edge_root_path = '/mnt/data/oss_beijing/qiank/Dataset/Benchmark Dataset/Train/'+ \
            net_type + '/' + data_type + ' ' + num

edge_paths = {'train': edge_root_path + '/Train_set.csv',
              'test': edge_root_path + '/Test_set.csv',
              'valid': edge_root_path + '/Validation_set.csv'}

data, split_edge = load_data(feat_path, edge_paths, args.use_pca, args.hidden_channels)

num_nodes, args.num_features = data.x.shape
directed = False

data_root = '/mnt/data/oss_beijing/qiank/pyg_datasets/' + data_name
if args.use_pca:
    data_root = data_root + '_pca'

if args.num_hops > 1:
    data_root = data_root + '_hop' + str(args.num_hops)

train_dataset, val_dataset, test_dataset = \
    LoadDatasets(args.dynamic, data_root, data, split_edge,
                 num_hops=args.num_hops, percent=args.train_percent,
                 node_label=args.node_label, ratio_per_hop=args.ratio_per_hop,
                 max_nodes_per_hop=args.max_nodes_per_hop,
                 directed=directed, split=['train', 'valid', 'test'])
