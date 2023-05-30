from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
import sys
import os.path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets'))
from scGNN import GENELink
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='mESC', help='cell type name')
parser.add_argument('--ground_truth', default='Specific', help='ground_truth name')
parser.add_argument('--varying_genes', default=500, help='number of genes')

parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 90, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type', type=str, default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction', type=str, default='concate', help='how to integrate multihead attention in 1st layer')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(args)


"""
Non-Specific
mHSC-L learning rate = 3e-5
"""
data_type = args.data_name
net_type = args.ground_truth
num = args.varying_genes



def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)



density = Network_Statistic(data_type,num,net_type)
exp_file = dataset_path+'/BEELINE_genelink/'+net_type+' Dataset/'+data_type+'/TFs+'+str(num)+'/BL--ExpressionData.csv'
tf_file = dataset_path+'/BEELINE_genelink/'+net_type+' Dataset/'+data_type+'/TFs+'+str(num)+'/TF.csv'
target_file = dataset_path+'/BEELINE_genelink/'+net_type+' Dataset/'+data_type+'/TFs+'+str(num)+'/Target.csv'

data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)

train_file = dataset_path+'/BEELINE_genelink/Train_validation_test/'+net_type+'/'+data_type+' '+str(num)+'/Train_set.csv'
test_file = dataset_path+'/BEELINE_genelink/Train_validation_test/'+net_type+'/'+data_type+' '+str(num)+'/Test_set.csv'
val_file = dataset_path+'/BEELINE_genelink/Train_validation_test/'+net_type+'/'+data_type+' '+str(num)+'/Validation_set.csv'


tf_embed_path = r'Result/'+net_type+'/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/'+net_type+'/'+data_type+' '+str(num)+'/Channel2.csv'
if not os.path.exists('Result/'+net_type+'/'+data_type+' '+str(num)):
    os.makedirs('Result/'+net_type+'/'+data_type+' '+str(num))

train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf,loop=args.loop)


adj = adj2saprse_tensor(adj)


train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

model = GENELink(input_dim=feature.size()[1],
                hidden1_dim=args.hidden_dim[0],  # ConvLayer1  --hidden_dim', default=[128,64,32]
                hidden2_dim=args.hidden_dim[1],  # ConvLayer2
                hidden3_dim=args.hidden_dim[2],  # Linear1
                output_dim=args.output_dim,      # Linear1    --output_dim', default=16
                num_head1=args.num_head[0],      # --num_head', default=[3,3]
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                )


adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)


optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = 'model/'+ net_type+'/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
figure_path = 'figures/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)


model.eval()
with torch.no_grad():
    pred = model(data_feature, adj, train_data)
    score = model(data_feature, adj, validation_data)
if args.flag:
    pre = torch.softmax(pred, dim=1)
    score = torch.softmax(score, dim=1)
else:
    pred = torch.sigmoid(pred)
    score = torch.sigmoid(score)
loss_BCE = F.binary_cross_entropy(pred, train_data[:, -1].float().view(-1, 1))
AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=val_data[:, -1],flag=args.flag)

print('(Before training) Validation AUC:{:.3F}'.format(AUC),
      'AUPR:{:.3F}'.format(AUPR))

for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)


        # train_y = train_y.to(device).view(-1, 1)
        pred = model(data_feature, adj, train_x)

        #pred = torch.sigmoid(pred)
        if args.flag:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        loss_BCE = F.binary_cross_entropy(pred, train_y)


        loss_BCE.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_BCE.item()


    model.eval()
    score = model(data_feature, adj, validation_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    # score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
        #
    print('Epoch:{}'.format(epoch + 1),
            'train loss:{}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))


torch.save(model.state_dict(), model_path + data_type+' '+str(num)+'.pkl')

model.load_state_dict(torch.load(model_path + data_type+' '+str(num)+'.pkl'))
model.eval()
tf_embed, target_embed = model.get_embedding()
embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)


score = model(data_feature, adj, test_data)
if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)
# score = torch.sigmoid(score)


AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)

print('Net type:{}, Data: {} {}, AUC:{:.3F}, AUPRC:{:.3F}'.format(net_type, data_type, num, AUC, AUPR))






















