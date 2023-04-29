# AI4GRN





## System requirement

#### Programming language
Python 3.5 +

#### Python Packages
Pytorch , Numpy , Networkx, PyTorch Geometric

## Training 

#### Train the **LGLP** to compare with GENELink

```
cd projects/LGLP-main/Python
python Main_vs_genelink.py --data-name hESC --ground_truth STRING --varying_genes 500
```

#### Train the **SEAL** to compare with GENELink

```
cd projects/SEAL/Python
python Main_vs_genelink.py --data-name hESC --ground_truth STRING --varying_genes 500
python Main_vs_genelink.py --data-name hESC --ground_truth STRING --varying_genes 500 --use-attribute
```
