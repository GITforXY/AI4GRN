import torch
import numpy as np
import pandas as pd
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets'))
import random
import argparse
from torch.utils.data import DataLoader
sys.path.append('%s/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *


parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data_name', default=None, help='cell type name')
parser.add_argument('--ground_truth', default=None, help='ground_truth name')
parser.add_argument('--varying_genes', default=None, help='number of genes')

parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; \
                    sample a portion from them as negative training data. Otherwise,\
                    train negative and test negative data are both sampled from \
                    unknown links without overlap.')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='save the final model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
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

# check whether train and test links are provided
train_pos, test_pos = None, None
if args.train_name is not None:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    train_pos = (train_idx[:, 0], train_idx[:, 1])
if args.test_name is not None:
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    test_pos = (test_idx[:, 0], test_idx[:, 1])

# build observed network
if args.data_name is not None:  # use .mat network
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    # data = sio.loadmat(args.data_dir)
    data = sio.loadmat(
        dataset_path+'/BEELINE_genelink/'+args.ground_truth+' Dataset/'+args.data_name+'/TFs+'+args.varying_genes+'/input.mat')
    net = data['net']
    if 'group' in data:
        # load node attributes (here a.k.a. node classes)
        attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
else:  # build network from train links
    assert (args.train_name is not None), "must provide train links if not using .mat"
    if args.train_name.endswith('_train.txt'):
        args.data_name = args.train_name[:-10] 
    else:
        args.data_name = args.train_name.split('.')[0]
    max_idx = np.max(train_idx)
    if args.test_name is not None:
        max_idx = max(max_idx, np.max(test_idx))
    net = ssp.csc_matrix(
        (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), 
        shape=(max_idx+1, max_idx+1)
    )
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

# sample train and test links
if args.train_name is None and args.test_name is None:
    # sample both positive and negative train/test links from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net,
        args.test_ratio,
        max_train_num=args.max_train_num
    )
else:
    # use provided train/test positive links, sample negative from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, 
        train_pos=train_pos, 
        test_pos=test_pos, 
        max_train_num=args.max_train_num,
        all_unknown_as_negative=args.all_unknown_as_negative
    )


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
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

if args.only_predict:  # no need to use negatives
    _, test_graphs, max_n_label = links2subgraphs(
        A, 
        None, 
        None, 
        test_pos,  # test_pos is a name only, we don't actually know their labels
        None, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel
    )
    print('# test: %d' % (len(test_graphs)))
else:
    train_graphs, test_graphs, val_graphs, max_n_label = links2subgraphs(
        A, 
        train_pos, 
        train_neg, 
        test_pos, 
        test_neg, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel,
        val_pos,
        val_neg
    )
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

# DGCNN configurations
if args.only_predict:
    with open('data/{}_hyper.pkl'.format(args.data_name), 'rb') as hyperparameters_name:
        saved_cmd_args = pickle.load(hyperparameters_name)
    for key, value in vars(saved_cmd_args).items(): # replace with saved cmd_args
        vars(cmd_args)[key] = value
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    model_name = 'data/{}_model.pth'.format(args.data_name)
    classifier.load_state_dict(torch.load(model_name))
    classifier.eval()
    predictions = []
    batch_graph = []
    for i, graph in enumerate(test_graphs):
        batch_graph.append(graph)
        if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
            predictions.append(classifier(batch_graph)[0][:, 1].exp().cpu().detach())
            batch_graph = []
    predictions = torch.cat(predictions, 0).unsqueeze(1).numpy()
    test_idx_and_pred = np.concatenate([test_idx, predictions], 1)
    pred_name = 'data/' + args.test_name.split('.')[0] + '_pred.txt'
    np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%d', '%1.2f'])
    print('Predictions for {} are saved in {}'.format(args.test_name, pred_name))
    exit()


cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
# cmd_args.num_epochs = 50
cmd_args.num_epochs = 20
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

# split o.1 val_set
# random.shuffle(train_graphs)
# val_num = int(0.1 * len(train_graphs))
# val_graphs = train_graphs[:val_num]
# train_graphs = train_graphs[val_num:]
train_metrics = np.zeros((cmd_args.num_epochs+1, 4))  # add initial_results
val_metrics = np.zeros((cmd_args.num_epochs+1, 4))
train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None

for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    if epoch == 0:
        classifier.eval()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
            epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3]))
        val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
        if not cmd_args.printAUC:
            val_loss[2] = 0.0
        print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
            epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3]))
        train_metrics[0] = np.array(avg_loss)
        val_metrics[0] = np.array(val_loss)

    classifier.train()
    avg_loss = loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size
    )
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3]))

    classifier.eval()
    val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
        epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3]))

    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch

    train_metrics[epoch + 1] = np.array(avg_loss)
    val_metrics[epoch + 1] = np.array(val_loss)

# todo: don't test when training
test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
if not cmd_args.printAUC:
    test_loss[2] = 0.0
print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
    epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))

plot_history(train_metrics, val_metrics,
             args.data_name, args.ground_truth, args.varying_genes, args.use_attribute)

if args.save_model:
    # model_name = 'data/{}_model.pth'.format(args.data_name)
    model_name = dataset_path+'/BEELINE_genelink/Train_validation_test/'+\
                 args.ground_truth+'/'+args.data_name+' '+args.varying_genes+\
                 '/SEAL_attribute_{}_model.pth'.format(args.use_attribute)
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    hyper_name = dataset_path+'/BEELINE_genelink/Train_validation_test/'+\
                 args.ground_truth+'/'+args.data_name+' '+args.varying_genes+\
                 '/SEAL_attribute_{}_hyper.pkl'.format(args.use_attribute)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters to {}...'.format(hyper_name))

with open('ap_results.txt', 'a+') as f:
    f.write(args.data_name+' '+args.ground_truth+' '+args.varying_genes+' attribute_'+str(args.use_attribute) +
            str(test_loss[3]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(args.data_name+' '+args.ground_truth+' '+args.varying_genes+' attribute_'+str(args.use_attribute) +
                str(test_loss[2]) + '\n')

