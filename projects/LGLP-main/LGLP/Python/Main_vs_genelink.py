import numpy
import torch
import numpy as np
import pandas as pd
import sys, copy, math, time, pdb
import pickle as pickle
import json
import scipy.io as sio
import scipy.sparse as ssp
import os.path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'))
import random
import argparse
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net


parser = argparse.ArgumentParser(description='Link Prediction')
# general settings
parser.add_argument('--data_name', default='BUP', help='network name')
parser.add_argument('--ground_truth', default=None, help='ground_truth name')
parser.add_argument('--varying_genes', default=None, help='number of genes')

parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
# parser.add_argument('--use-attributes', default=False, help='use-attributes')
parser.add_argument('--max-train-num', type=int, default=10000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.5,
                    help='ratio of test links')
# model settings
parser.add_argument('--hop', default=2, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed) 
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:

    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    # data = sio.loadmat(args.data_dir)
    data = sio.loadmat(
        dataset_path+'/BEELINE_genelink/' + args.ground_truth + ' Dataset/' + args.data_name + '/TFs+' + args.varying_genes + '/input.mat')

    net = data['net']  # Adjacency matrix
    print("# row of net: %d , # nonzero elements number: %d" % (net.shape[0], net.getnnz()))
    attributes = None
    # if 'group' in data:
    #     # load node attributes (here a.k.a. node classes)
    #     attributes = data['group']
    #     print("# row of attributes: %d , # nonzero elements number: %d" % (attributes.shape[0], attributes.getnnz()))
    #     attributes = attributes.toarray().astype('float32')

    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    # #Sample train and test links
    # train_pos, train_neg, test_pos, test_neg = sample_neg(
    #     net,
    #     args.test_ratio,
    #     max_train_num=args.max_train_num)
    # print('# train_pos: %d, # test_pos: %d' % (len(train_pos[0]), len(test_pos[0])))
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)


# sample train and test links by GENELINK
train_set = pd.read_csv(
    dataset_path+'/BEELINE_genelink/Train_validation_test/'+
    args.ground_truth+'/'+args.data_name+' '+args.varying_genes+'/Train_set.csv')
val_set = pd.read_csv(
    dataset_path+'/BEELINE_genelink/Train_validation_test/'+
    args.ground_truth+'/'+args.data_name+' '+args.varying_genes+'/Validation_set.csv')
test_set = pd.read_csv(
    dataset_path+'/BEELINE_genelink/Train_validation_test/'+
    args.ground_truth+'/'+args.data_name+' '+args.varying_genes+'/Test_set.csv')

train_pos = ([int(row['TF']) for index, row in train_set.iterrows() if row['Label'] == 1],
             [int(row['Target']) for index, row in train_set.iterrows() if row['Label'] == 1])
train_neg = ([int(row['TF']) for index, row in train_set.iterrows() if row['Label'] == 0],
             [int(row['Target']) for index, row in train_set.iterrows() if row['Label'] == 0])
val_pos = ([int(row['TF']) for index, row in val_set.iterrows() if row['Label'] == 1],
           [int(row['Target']) for index, row in val_set.iterrows() if row['Label'] == 1])
val_neg = ([int(row['TF']) for index, row in val_set.iterrows() if row['Label'] == 0],
           [int(row['Target']) for index, row in val_set.iterrows() if row['Label'] == 0])
test_pos = ([int(row['TF']) for index, row in test_set.iterrows() if row['Label'] == 1],
            [int(row['Target']) for index, row in test_set.iterrows() if row['Label'] == 1])
test_neg = ([int(row['TF']) for index, row in test_set.iterrows() if row['Label'] == 0],
            [int(row['Target']) for index, row in test_set.iterrows() if row['Label'] == 0])
print("train_pos:%d, train_neg:%d" % (len(train_pos[0]), len(train_neg[0])))
print("val_pos:%d, val_neg:%d" % (len(val_pos[0]), len(val_neg[0])))
print("test_pos:%d, test_neg:%d" % (len(test_pos[0]), len(test_neg[0])))


'''Train and apply classifier'''
A = net.copy()  # the observed network
# A[val_pos[0], val_pos[1]] = 0  # mask test links
# A[val_pos[1], val_pos[0]] = 0  # mask test links
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
A.eliminate_zeros()

# train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, None)
train_graphs, test_graphs, val_graphs, max_n_label = links2subgraphs(
    A,
    train_pos,
    train_neg,
    test_pos,
    test_neg,
    args.hop,
    args.max_nodes_per_hop,
    attributes,
    val_pos,
    val_neg
)
print(('# train subgraph: %d, # test subgraph: %d' % (len(train_graphs), len(test_graphs))))


train_lines = to_linegraphs(train_graphs, max_n_label)  # train_lines的维度和max_n_label的关系？
test_lines = to_linegraphs(test_graphs, max_n_label)  # train_lines.shape[1]=(max_n_label + 1)*2
val_lines = to_linegraphs(val_graphs, max_n_label)

print("Edge feature shape:", train_lines[5])
# train_lines[0]: Data(edge_index=[2, 54], num_nodes=11, x=[11, 20], y=[1])

# edge_index：划分的训练集(或验证集、测试集)中的所有边的端点标号，形状为(2, num_edges)
# edge_attr：划分的训练集(或验证集、测试集)中所有边的特征，形状为(num_edges, 3)
# x：划分的训练集(或验证集、测试集)中所有节点的特征，形状为(num_nodes, 9)
# y：该数据集中所有图的标签，形状为(num_graphs, 1)



# Model configurations

# cmd_args.latent_dim = [32, 32, 32, 1]
# cmd_args.hidden = 128
cmd_args.latent_dim = [32, 32, 32, 1]  # 改变图卷积层数
cmd_args.hidden = 128
cmd_args.out_dim = 0
# cmd_args.dropout = True
cmd_args.dropout = False
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
# cmd_args.num_epochs = 15
cmd_args.num_epochs = 20
cmd_args.learning_rate = 5e-3
# cmd_args.learning_rate = 5e-5  #　改变lr
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = (max_n_label + 1)*2  # linegraph中的1个node包含原graph的2个node的信息
# cmd_args.feat_dim = (max_n_label + 1)
cmd_args.attr_dim = 0
# if attributes is not None:
#     cmd_args.attr_dim = attributes.shape[1]*2
#     cmd_args.feat_dim = cmd_args.feat_dim + cmd_args.attr_dim

train_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=True)
test_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)
val_loader = DataLoader(val_lines, batch_size=cmd_args.batch_size, shuffle=False)



classifier = Net(
    cmd_args.feat_dim,
    # cmd_args.feat_dim + cmd_args.attr_dim,
    cmd_args.hidden,
    cmd_args.latent_dim,
    cmd_args.dropout)
if cmd_args.mode == 'gpu':
    classifier = classifier.to("cuda")

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)



best_auc = 0
best_auc_acc = 0
best_acc = 0
best_acc_auc = 0
losses_train, ap_train, auc_train = [], [], []
losses_test, ap_test, auc_test = [], [], []

train_metrics = np.zeros((cmd_args.num_epochs+1, 4))  # add initial_results
val_metrics = np.zeros((cmd_args.num_epochs+1, 4))

best_loss = None
best_epoch = None

for epoch in range(cmd_args.num_epochs):
    if epoch == 0:
        classifier.eval()
        train_loss, _, initial_state_dict = loop_dataset_gem(classifier, train_loader, None)
        if not cmd_args.printAUC:
            train_loss[2] = 0.0
        print(('\033[92m(Initial) average training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3])))
        val_loss, _, _ = loop_dataset_gem(classifier, val_loader, None)
        if not cmd_args.printAUC:
            train_loss[2] = 0.0
        print(('\033[92m(Initial) average validation of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
        epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3])))
        train_metrics[0] = np.array(train_loss)
        val_metrics[0] = np.array(val_loss)

    classifier.train()
    avg_loss, batch_loss, _ = loop_dataset_gem(classifier, train_loader, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' %
           (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))

    classifier.eval()
    # TODO: calculate losses for train datasets after each epoch's training
    # train_loss, _, _= loop_dataset_gem(classifier, train_loader, None)
    # if not cmd_args.printAUC:
    #     train_loss[2] = 0.0
    # print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' %
    # (epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3])))
    val_loss, _, _ = loop_dataset_gem(classifier, val_loader, None)
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
        epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3]))

    train_metrics[epoch+1] = np.array(avg_loss)
    val_metrics[epoch+1] = np.array(val_loss)

    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch

    if epoch == 0:
        metrics_name = ['loss', 'acc', 'auc', 'ap']
        for i in range(4):
            plt.plot(batch_loss[:, i], label=metrics_name[i])
            plt.title(metrics_name[i], fontsize=20)
            plt.xlabel('batch')
            plt.legend()
            plt.savefig('./figures/'+args.data_name+'_batch' + '_' + metrics_name[i] + '.png')
            plt.clf()

test_loss, _, _ = loop_dataset_gem(classifier, test_loader, None)
if not cmd_args.printAUC:
    test_loss[2] = 0.0
print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
    epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))


plot_history(train_metrics, val_metrics,
             args.data_name, args.ground_truth, args.varying_genes)
# print("best_auc:", best_auc)
# print("best_ap", best_acc)

# model_name = 'data/{}_model.pth'.format(args.data_name)
model_name = dataset_path+'/BEELINE_genelink/Train_validation_test/'+\
             args.ground_truth+'/'+args.data_name+' '+args.varying_genes+\
             '/LGLP_model.pth'
print('Saving final model states to {}...'.format(model_name))
torch.save(classifier.state_dict(), model_name)
hyper_name = dataset_path+'/BEELINE_genelink/Train_validation_test/'+\
             args.ground_truth+'/'+args.data_name+' '+args.varying_genes+\
             '/LGLP_hyper.pkl'
with open(hyper_name, 'wb') as hyperparameters_file:
    pickle.dump(cmd_args, hyperparameters_file)
    print('Saving hyperparameters to {}...'.format(hyper_name))

with open('ap_results.txt', 'a+') as f:
    f.write(args.data_name+' '+args.ground_truth+' '+args.varying_genes+' '+
            str(test_loss[3]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(args.data_name+' '+args.ground_truth+' '+args.varying_genes+' '+
                str(test_loss[2]) + '\n')
