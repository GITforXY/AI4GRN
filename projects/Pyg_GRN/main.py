import argparse
import os
from data import load_data
import math
import torch
from data import LoadDatasets
from torch_geometric.data import DataLoader
from train import LP_GRN
import time
from torch.utils.tensorboard import SummaryWriter
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
# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--use_ignn', action='store_true', help="whether to use ignn")
parser.add_argument('--use_gatv2', action='store_true', help="whether to use gatv2")
parser.add_argument('--pre_trained', action='store_true',
                    help="if provided pre-trained model, only train other parameters")
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)

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
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--dynamic', action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--num_workers', type=int, default=4,
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                    help="load pretrained node embeddings as additional node features")

# Testing settings
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=5)
parser.add_argument('--only_test', action='store_true',
                    help="only test without training")
parser.add_argument('--test_multiple_models', action='store_true',
                    help="test multiple models together")

args = parser.parse_args()

if args.use_pca:
    args.use_feature = True
# Load data
data_type, net_type, num = args.data, args.net, str(args.num)
data_name = net_type + '_' + data_type + '_' + num

if args.train_node_embedding:
    args.res_dir = os.path.join('results/{}_{}'.format(args.model, 'emb'))
else:
    args.res_dir = os.path.join('results/{}_{}'.format(args.model, 'noemb'))

print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          pin_memory=True, prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                         num_workers=args.num_workers, pin_memory=True)

## args

args.max_z = 1000

if 'DGCNN' in args.model:
    if args.sortpool_k <= 1:  # Transform percentile to number.
        if args.dynamic:
            sampled_train = train_dataset[:1000]
        else:
            sampled_train = train_dataset
        subg_num_nodes = sorted([g.num_nodes for g in sampled_train])
        k = subg_num_nodes[int(math.ceil(args.sortpool_k * len(subg_num_nodes))) - 1]
        args.sortpool_k = max(10, k)

    print(f'SortPooling k is set to {args.sortpool_k}')

# Training starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = LP_GRN(args, eval_metric=eval, device=device, num_nodes=num_nodes,
                 num_data=len(train_dataset))
writer = SummaryWriter('/runs')

best_val_epoch = 0
best_val_acu = 0.0
all_test_auc = []

for epoch in range(args.n_epochs):
    T1 = time.time()

    train_loss = trainer.train(train_loader=train_loader)
    val_loss, val_auc = trainer.val(val_loader=val_loader)
    print('Epoch: {}, Train Loss: {}, Valid Loss: {}, Valid auc: {}'.format(epoch+1, train_loss, val_loss, val_auc))

    if (epoch+1) >= 2 and val_auc > best_val_acu:
        best_val_epoch = epoch
        best_val_acu = val_auc

    test_auc = trainer.test(test_loader=test_loader)
    print('Test auc: {}'.format(test_auc))
    all_test_auc.append(test_auc)

    T2 = time.time()
    print('Time used: %s second' % (T2 - T1))

    writer.add_scalars(data_type + '_' + net_type + '_' + num + '_loss',
                       {'train': train_loss, 'val': val_loss}, epoch)
    writer.add_scalars(data_type + '_' + net_type + '_' + num + '_auc',
                       {'val': val_auc, 'test': test_auc}, epoch)


writer.close()
model_name = os.path.join(args.res_dir, '{}_{}_checkpoint.pth'.format(data_name, str(args.num_hops)))
torch.save(trainer.model.state_dict(), model_name)

print("Final test auc for data '{}' is: {}, for best_val is: {}, best test is: {}".format(
    data_name, all_test_auc[-1], all_test_auc[best_val_epoch], max(all_test_auc)))
