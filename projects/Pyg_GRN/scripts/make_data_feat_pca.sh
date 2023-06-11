#!/bin/bash

#### TF + 500 genes
## STRING
python make_data.py --num 500 --data hESC --net STRING --use_pca
python make_data.py --num 500 --data hHEP --net STRING --use_pca
python make_data.py --num 500 --data mDC --net STRING --use_pca
python make_data.py --num 500 --data mESC --net STRING --use_pca
python make_data.py --num 500 --data mHSC-E --net STRING --use_pca
python make_data.py --num 500 --data mHSC-GM --net STRING --use_pca
python make_data.py --num 500 --data mHSC-L --net STRING --use_pca

## Non-specific
python make_data.py --num 500 --data hESC --net Non-Specific --use_pca
python make_data.py --num 500 --data hHEP --net Non-Specific --use_pca
python make_data.py --num 500 --data mDC --net Non-Specific --use_pca
python make_data.py --num 500 --data mESC --net Non-Specific --use_pca
python make_data.py --num 500 --data mHSC-E --net Non-Specific --use_pca
python make_data.py --num 500 --data mHSC-GM --net Non-Specific --use_pca
python make_data.py --num 500 --data mHSC-L --net Non-Specific --use_pca

## Lofgof
python make_data.py --num 500 --data mESC --net Lofgof --use_pca

## Specific
python make_data.py --num 500 --data hESC --net Specific --use_pca
python make_data.py --num 500 --data hHEP --net Specific --use_pca
python make_data.py --num 500 --data mDC --net Specific --use_pca
python make_data.py --num 500 --data mESC --net Specific --use_pca
python make_data.py --num 500 --data mHSC-E --net Specific --use_pca
python make_data.py --num 500 --data mHSC-GM --net Specific --use_pca
python make_data.py --num 500 --data mHSC-L --net Specific --use_pca


##### TF + 1000 genes

## STRING
python make_data.py --num 1000 --data hESC --net STRING --use_pca
python make_data.py --num 1000 --data hHEP --net STRING --use_pca
python make_data.py --num 1000 --data mDC --net STRING --use_pca
python make_data.py --num 1000 --data mESC --net STRING --use_pca
python make_data.py --num 1000 --data mHSC-E --net STRING --use_pca
python make_data.py --num 1000 --data mHSC-GM --net STRING --use_pca
python make_data.py --num 1000 --data mHSC-L --net STRING --use_pca

## Non-specific
python make_data.py --num 1000 --data hESC --net Non-Specific --use_pca
python make_data.py --num 1000 --data hHEP --net Non-Specific --use_pca
python make_data.py --num 1000 --data mDC --net Non-Specific --use_pca
python make_data.py --num 1000 --data mESC --net Non-Specific --use_pca
python make_data.py --num 1000 --data mHSC-E --net Non-Specific --use_pca
python make_data.py --num 1000 --data mHSC-GM --net Non-Specific --use_pca
python make_data.py --num 1000 --data mHSC-L --net Non-Specific --use_pca

## Lofgof
python make_data.py --num 1000 --data mESC --net Lofgof --use_pca

## Specific
python make_data.py --num 1000 --data hESC --net Specific --use_pca
python make_data.py --num 1000 --data hHEP --net Specific --use_pca
python make_data.py --num 1000 --data mDC --net Specific --use_pca
python make_data.py --num 1000 --data mESC --net Specific --use_pca
python make_data.py --num 1000 --data mHSC-E --net Specific --use_pca
python make_data.py --num 1000 --data mHSC-GM --net Specific --use_pca
python make_data.py --num 1000 --data mHSC-L --net Specific --use_pca

