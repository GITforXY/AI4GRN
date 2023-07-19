# AI4GRN

## System requirement

### 个人镜像
镜像地址：pjlab-3090-registry-vpc.cn-beijing.cr.aliyuncs.com/ai4bio/xieyong:ai4grn-xieyong
```
conda activate              # 可运行 GENIE3, SEACells
conda activate GRGNN        # 可运行 SEAL_pyg, GENELink, SERGIO
conda activate graphmae     # 可运行 GraphMAE
```
```
cd /mnt/workspace/xieyong/projects/
pip install -r requirements_base.txt
pip install -r requirements_GRGNN.txt
pip install -r requirements_graphmae.txt
```
## dataset & preprocessing
### beeline数据集：
表达特征目录：`/mnt/data/oss_beijing/xieyong/datasets/BEELINE_genelink`

数据划分目录：`/mnt/data/oss_beijing/qiank/Dataset/Benchmark Dataset/Train1_1/`

保存pyg子图目录：`/mnt/data/oss_beijing/xieyong/pyg_dataset`

保存pyg子图目录：`/mnt/data/oss_beijing/xieyong/pyg_new`

数据划分代码：`/mnt/data/oss_beijing/qiank/Dataset/Benchmark Dataset/Train_Test_Split1_1.py`

### human数据集：
表达特征目录：`/mnt/data/oss_beijing/xieyong/datasets/model_data/Expressions/`

数据划分目录：`/mnt/data/oss_beijing/xieyong/datasets/model_data/Data_Split_2qiank/`

提取表达特征/调控网络代码：`/mnt/workspace/qiank/data/prepare_network.ipynb`

数据划分代码：`/mnt/workspace/xieyong/projects/GENELink/Code/Train_Test_Split_model_data.py`

## Training 

### Train the **GENIE3** 
```
cd /mnt/workspace/xieyong/projects/GENIE3
bash run_beeline.sh     # beeline 数据
bash evaluate_beeline.sh

bash run_model_data.sh      # human 数据
bash evaluate_model_data.sh
```

### Train the **GENELink** 
```
cd /mnt/workspace/xieyong/projects/GENELink/Code
bash train.sh       # beeline 数据
bash train_model_data.sh      # human 数据
```

### Train the **SEAL_pyg** 
```
cd /mnt/workspace/xieyong/projects/Pyg_GRN/Python
python main_model_data.py --data AST --model DGCNN --use_feature --train_node_embedding --use_gatv2 --use_log --dynamic
```
`--model`: 默认模型为DGDNN 

`--use_feature`: 使用基因表达特征

`--train_node_embedding`：训练node_embedding

`--use_gatv2`：图卷积模块改用GAT_v2，默认GCN

`--use_log`：对count矩阵做log变换

`--dynamic`：每次重新提取子图

`--pre_trained`：加载预训练模型finetune

### Train the **SERGIO** 
```
cd /mnt/workspace/xieyong/projects/SERGIO
python run_sergio.py
```
```
sim = sergio(number_genes=1200, number_bins = 9, number_sc = 300, noise_params = 1,
             decays=0.8, sampling_state=15, noise_type='dpd')
```
`number_bins`：细胞类型的数目

`number_sc`：每个细胞类型的细胞数目

`noise_params`：添加噪音水平


### Train the **GraphMAE**

```
cd /mnt/workspace/xieyong/projects/GraphMAE
bash run_pretrain.sh
bash run_finetune.sh
```
`--dataset_name`：dgl子图存放目录下保存的子图

### Train the **SEACells**
```
cd /mnt/workspace/xieyong/projects/SEACells/notebooks
```
`SEACell_modeldata_computations.ipynb` 中的核心参数：

`sc.pp.highly_variable_genes(ad, n_top_genes=1500)`，选择高变基因

`sc.tl.pca(ad, n_comps=50, use_highly_variable=True)`，pca降维后用于模型输入

`n_SEACells = 32`，输出的metacells的数量
