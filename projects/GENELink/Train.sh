#!/bin/bash

#### TF + 500 genes
## STRING
python Train.py --num 500 --data hESC --net STRING
python Train.py --num 500 --data hHEP --net STRING
python Train.py --num 500 --data mDC --net STRING
python Train.py --num 500 --data mESC --net STRING
python Train.py --num 500 --data mHSC-E --net STRING
python Train.py --num 500 --data mHSC-GM --net STRING
python Train.py --num 500 --data mHSC-L --net STRING

## Non-specific
python Train.py --num 500 --data hESC --net Non-Specific
python Train.py --num 500 --data hHEP --net Non-Specific
python Train.py --num 500 --data mDC --net Non-Specific
python Train.py --num 500 --data mESC --net Non-Specific
python Train.py --num 500 --data mHSC-E --net Non-Specific
python Train.py --num 500 --data mHSC-GM --net Non-Specific
python Train.py --num 500 --data mHSC-L --net Non-Specific

## Lofgof
python Train.py --num 500 --data mESC --net Lofgof

## Specific
python Train.py --num 500 --data hESC --net Specific
python Train.py --num 500 --data hHEP --net Specific
python Train.py --num 500 --data mDC --net Specific
python Train.py --num 500 --data mESC --net Specific
python Train.py --num 500 --data mHSC-E --net Specific
python Train.py --num 500 --data mHSC-GM --net Specific
python Train.py --num 500 --data mHSC-L --net Specific


##### TF + 1000 genes

## STRING
python Train.py --num 1000 --data hESC --net STRING
python Train.py --num 1000 --data hHEP --net STRING
python Train.py --num 1000 --data mDC --net STRING
python Train.py --num 1000 --data mESC --net STRING
python Train.py --num 1000 --data mHSC-E --net STRING
python Train.py --num 1000 --data mHSC-GM --net STRING
python Train.py --num 1000 --data mHSC-L --net STRING

## Non-specific
python Train.py --num 1000 --data hESC --net Non-Specific
python Train.py --num 1000 --data hHEP --net Non-Specific
python Train.py --num 1000 --data mDC --net Non-Specific
python Train.py --num 1000 --data mESC --net Non-Specific
python Train.py --num 1000 --data mHSC-E --net Non-Specific
python Train.py --num 1000 --data mHSC-GM --net Non-Specific
python Train.py --num 1000 --data mHSC-L --net Non-Specific

## Lofgof
python Train.py --num 1000 --data mESC --net Lofgof

## Specific
python Train.py --num 1000 --data hESC --net Specific
python Train.py --num 1000 --data hHEP --net Specific
python Train.py --num 1000 --data mDC --net Specific
python Train.py --num 1000 --data mESC --net Specific
python Train.py --num 1000 --data mHSC-E --net Specific
python Train.py --num 1000 --data mHSC-GM --net Specific
python Train.py --num 1000 --data mHSC-L --net Specific

