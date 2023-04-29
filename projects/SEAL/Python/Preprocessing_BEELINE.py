import time
import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
from scipy import interpolate
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import scipy.sparse
import scipy.io as sio
import sys
import pickle

# Preprocess DREAM5 data from Official DREAM websites.
# https://www.synapse.org/#!Synapse:syn3130840
parser = argparse.ArgumentParser()

parser.add_argument('--ground_truth', type=str, default='Specific')
parser.add_argument('--cell_type', type=str, default='hESC')
parser.add_argument('--varying_genes', type=str, default='500')

args = parser.parse_args()

# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, sample_size, feature_size):
    samplelist=[]
    featurelist=[]
    data =[]
    count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if count >= 0:
                line = line.strip()
                words = line.split(',')  # 同一个sample中的不同gene的表达
                data_count = 0
                for word in words[1:]:
                    featurelist.append(count)  # [0,0...,0,0, 1,1,...1,1, 2,2...2,2, 3,3,...]
                    samplelist.append(data_count)  # [0,1,2,3...,num_cols, 0,1,2,3...num_genes, 0,1,2...]
                    data.append(float(word))
                    data_count += 1
            count += 1
    f.close()
    feature = scipy.sparse.csr_matrix((data, (featurelist, samplelist)), shape=(feature_size, sample_size,))
    return feature

# Load gold standard edges into sparse matrix
def read_edge_file_csc(filename, feature_size):
    row=[]
    col=[]
    data=[]
    count = -1  # 记录已读边的数量
    with open(filename) as f:
        lines = f.readlines()  # 读取每行
        for line in lines:
            if count >= 0:
                line = line.strip()  # 去除开头结尾的空格
                words = line.split(',')  # 拆分成单词
                end1 = int(words[1])  # 例：words[0]=='G2',代表第2个节点,end1==1，代表索引
                end2 = int(words[2])
                if end1 > end2:  # 排序，节点ID由小到大
                    tmpp = end1
                    end1 = end2
                    end2 = tmpp
                row.append(end1)
                col.append(end2)
                data.append(1.0)
                row.append(end2)
                col.append(end1)
                data.append(1.0)
            count += 1
    f.close()
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(feature_size, feature_size))
    return mtx


ground_truth = args.ground_truth
cell_type = args.cell_type
varying_genes = args.varying_genes
# feature_filename = "/nvme/xieyong/projects/GRGNN-master/data/DREAM5_network_inference_challenge/Network"+datasetname+"/input_data/net"+datasetname+"_expression_data.tsv"
# edge_filename    = "/nvme/xieyong/projects/GRGNN-master/data/DREAM5_network_inference_challenge/Network"+datasetname+"/gold_standard/DREAM5_NetworkInference_GoldStandard_Network"+datasetname+".tsv"
feature_filename = "/nvme/xieyong/datasets/BEELINE_genelink/"+ground_truth+" Dataset/"+cell_type+"/TFs+"+varying_genes+"/BL--ExpressionData.csv"
edge_filename    = "/nvme/xieyong/datasets/BEELINE_genelink/"+ground_truth+" Dataset/"+cell_type+"/TFs+"+varying_genes+"/Label.csv"

df = pd.read_csv(feature_filename, index_col=0)
feature_size = df.shape[0]
sample_size = df.shape[1]
print('ExpressionData features: %d, samples: %d' % (feature_size, sample_size))

graphcsc = read_edge_file_csc(edge_filename, feature_size=feature_size)
print('Adjacent Matrix size:', graphcsc.shape)
allx = read_feature_file_sparse(feature_filename, sample_size=sample_size, feature_size=feature_size)
print('Features Matrix size:', allx.shape)

# # 保存为 .mat 文件
sio.savemat("/nvme/xieyong/datasets/BEELINE_genelink/"+ground_truth+" Dataset/"+cell_type+"/TFs+"+varying_genes+"/input.mat", {'net': graphcsc, 'group': allx})

# pickle.dump(allx, open( "ind.dream"+datasetname+".allx", "wb" ) )
# pickle.dump(graphcsc, open( "ind.dream"+datasetname+".csc", "wb" ) )

