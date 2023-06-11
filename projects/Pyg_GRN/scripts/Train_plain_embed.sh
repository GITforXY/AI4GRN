#!/bin/bash

#### TF + 500 genes
## STRING
python main.py --num 500 --data hESC --net STRING --train_node_embedding
python main.py --num 500 --data hHEP --net STRING --train_node_embedding
python main.py --num 500 --data mDC --net STRING --train_node_embedding
python main.py --num 500 --data mESC --net STRING --train_node_embedding
python main.py --num 500 --data mHSC-E --net STRING --train_node_embedding
python main.py --num 500 --data mHSC-GM --net STRING --train_node_embedding
python main.py --num 500 --data mHSC-L --net STRING --train_node_embedding

## Non-specific
python main.py --num 500 --data hESC --net Non-Specific --train_node_embedding
python main.py --num 500 --data hHEP --net Non-Specific --train_node_embedding
python main.py --num 500 --data mDC --net Non-Specific --train_node_embedding
python main.py --num 500 --data mESC --net Non-Specific --train_node_embedding
python main.py --num 500 --data mHSC-E --net Non-Specific --train_node_embedding
python main.py --num 500 --data mHSC-GM --net Non-Specific --train_node_embedding
python main.py --num 500 --data mHSC-L --net Non-Specific --train_node_embedding

## Lofgof
python main.py --num 500 --data mESC --net Lofgof --train_node_embedding

## Specific
python main.py --num 500 --data hESC --net Specific --train_node_embedding
python main.py --num 500 --data hHEP --net Specific --train_node_embedding
python main.py --num 500 --data mDC --net Specific --train_node_embedding
python main.py --num 500 --data mESC --net Specific --train_node_embedding
python main.py --num 500 --data mHSC-E --net Specific --train_node_embedding
python main.py --num 500 --data mHSC-GM --net Specific --train_node_embedding
python main.py --num 500 --data mHSC-L --net Specific --train_node_embedding


##### TF + 1000 genes

## STRING
python main.py --num 1000 --data hESC --net STRING --train_node_embedding
python main.py --num 1000 --data hHEP --net STRING --train_node_embedding
python main.py --num 1000 --data mDC --net STRING --train_node_embedding
python main.py --num 1000 --data mESC --net STRING --train_node_embedding
python main.py --num 1000 --data mHSC-E --net STRING --train_node_embedding
python main.py --num 1000 --data mHSC-GM --net STRING --train_node_embedding
python main.py --num 1000 --data mHSC-L --net STRING --train_node_embedding

## Non-specific
python main.py --num 1000 --data hESC --net Non-Specific --train_node_embedding
python main.py --num 1000 --data hHEP --net Non-Specific --train_node_embedding
python main.py --num 1000 --data mDC --net Non-Specific --train_node_embedding
python main.py --num 1000 --data mESC --net Non-Specific --train_node_embedding
python main.py --num 1000 --data mHSC-E --net Non-Specific --train_node_embedding
python main.py --num 1000 --data mHSC-GM --net Non-Specific --train_node_embedding
python main.py --num 1000 --data mHSC-L --net Non-Specific --train_node_embedding

## Lofgof
python main.py --num 1000 --data mESC --net Lofgof --train_node_embedding

## Specific
python main.py --num 1000 --data hESC --net Specific --train_node_embedding
python main.py --num 1000 --data hHEP --net Specific --train_node_embedding
python main.py --num 1000 --data mDC --net Specific --train_node_embedding
python main.py --num 1000 --data mESC --net Specific --train_node_embedding
python main.py --num 1000 --data mHSC-E --net Specific --train_node_embedding
python main.py --num 1000 --data mHSC-GM --net Specific --train_node_embedding
python main.py --num 1000 --data mHSC-L --net Specific --train_node_embedding

