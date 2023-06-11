#!/bin/bash

#### TF + 500 genes
## STRING
python main.py --num 500 --data hESC --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data hHEP --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mDC --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mESC --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-E --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-GM --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-L --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

## Non-specific
python main.py --num 500 --data hESC --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data hHEP --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mDC --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mESC --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-E --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-GM --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-L --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

## Lofgof
python main.py --num 500 --data mESC --net Lofgof --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

## Specific
python main.py --num 500 --data hESC --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data hHEP --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mDC --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mESC --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-E --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-GM --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 500 --data mHSC-L --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca


##### TF + 1000 genes

## STRING
python main.py --num 1000 --data hESC --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data hHEP --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mDC --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mESC --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-E --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-GM --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-L --net STRING --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

## Non-specific
python main.py --num 1000 --data hESC --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data hHEP --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mDC --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mESC --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-E --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-GM --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-L --net Non-Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

## Lofgof
python main.py --num 1000 --data mESC --net Lofgof --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

## Specific
python main.py --num 1000 --data hESC --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data hHEP --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mDC --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mESC --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-E --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-GM --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca
python main.py --num 1000 --data mHSC-L --net Specific --model DGCNN --n_epochs 50 --train_node_embedding --use_pca

