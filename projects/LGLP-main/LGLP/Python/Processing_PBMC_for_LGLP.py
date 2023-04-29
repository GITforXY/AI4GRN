
import numpy as np
import scipy.io
import scipy.sparse
import pickle

'''查看文件属性'''
# adata = sc.read('G:\XY\实习\SH_AI_lab\数据集\Rdata_ATAC-SEQ_v1/blood_sparse.mtx')
# print(len(adata))
# print(len(adata[0]))
#
# data = adata.X   # ATAC表达矩阵
# print('data rows ', adata.shape[0])
# print('data columns', adata.shape[1])
# print('data[100]: ', data[100])
# import scipy.sparse as sp
# # sp.save_npz('sparse_matrix.npz', data)
#
# import numpy as np
#
# dense_data = data.toarray()  # Convert sparse matrix to dense matrix
# print('dense_data rows', dense_data.shape[0])
# print('dense_data columns', dense_data.shape[1])
# # np.savetxt('atac_expression_matrix.tsv', dense_data, delimiter='\t')

'''.mtx to .mat'''
# 读取 .mtx 文件
# matrix = scipy.io.mmread('/nvme/xieyong/datasets/Rdata_ATAC-SEQ_v1/blood_sparse.mtx')  # 23127 * 10250
matrix = scipy.io.mmread('/nvme/xieyong/datasets/Rdata_ATAC-SEQ_v1/scRNA-seq/pbmc_scRNA_matrix.mtx')  # 20287 * 35582
print(type(matrix))                  # <class 'scipy.sparse._coo.coo_matrix'>

# 转换为稀疏矩阵类型
sparse_matrix = scipy.sparse.csc_matrix(matrix)
print(type(sparse_matrix))           # <class 'scipy.sparse._csc.csc_matrix'>
print('data rows ', sparse_matrix.shape[0])   # 23127
print('data columns', sparse_matrix.shape[1])   # 10250


'''.csc to .mat'''
def read_edge_file_csc(filename, sample_size):
    row=[]
    col=[]
    data=[]
    count = 0  # 记录已读边的数量
    with open(filename) as f:
        lines = f.readlines()  # 读取每行
        for line in lines:
            line = line.strip()  # 去除开头结尾的空格
            words = line.split()  # 拆分成单词
            end1 = int(words[0])  # 例：words[0]=='68',代表第69个节点，则end1==68，代表索引
            end2 = int(words[1])
            if end1 > end2:  # 排序，节点ID由小到大
                tmpp = end1
                end1 = end2
                end2 = tmpp
            if end1 == end2:
                continue
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
    print("total links: ", len(row)/2)  # 10609
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(sample_size, sample_size))
    return mtx

edge_filename = '/nvme/xieyong/datasets/Rdata_ATAC-SEQ_v1/scRNA-seq/gold_standard_RNA.txt'
csc = read_edge_file_csc(edge_filename, sample_size=sparse_matrix.shape[0])
# with open('/nvme/xieyong/datasets/Rdata_ATAC-SEQ_v1/PBMC.csc', "rb") as f:
#     csc = pickle.load(f)
print(type(csc))                   # class 'scipy.sparse._csc.csc_matrix'>
print('data rows ', csc.shape[0])
print('data columns', csc.shape[1])

# # 保存为 .mat 文件
scipy.io.savemat('/nvme/xieyong/LGLP-main/LGLP/Python/data/PBMC_RNA.mat', {'net': csc, 'group': sparse_matrix})
