#!/bin/bash

#### TF + 500 genes
## STRING
python main.py --num 500 --data hESC --net STRING
python main.py --num 500 --data hHEP --net STRING
python main.py --num 500 --data mDC --net STRING
python main.py --num 500 --data mESC --net STRING
python main.py --num 500 --data mHSC-E --net STRING
python main.py --num 500 --data mHSC-GM --net STRING
python main.py --num 500 --data mHSC-L --net STRING

## Non-specific
python main.py --num 500 --data hESC --net Non-Specific
python main.py --num 500 --data hHEP --net Non-Specific
python main.py --num 500 --data mDC --net Non-Specific
python main.py --num 500 --data mESC --net Non-Specific
python main.py --num 500 --data mHSC-E --net Non-Specific
python main.py --num 500 --data mHSC-GM --net Non-Specific
python main.py --num 500 --data mHSC-L --net Non-Specific

## Lofgof
python main.py --num 500 --data mESC --net Lofgof

## Specific
python main.py --num 500 --data hESC --net Specific
python main.py --num 500 --data hHEP --net Specific
python main.py --num 500 --data mDC --net Specific
python main.py --num 500 --data mESC --net Specific
python main.py --num 500 --data mHSC-E --net Specific
python main.py --num 500 --data mHSC-GM --net Specific
python main.py --num 500 --data mHSC-L --net Specific


##### TF + 1000 genes

## STRING
python main.py --num 1000 --data hESC --net STRING
python main.py --num 1000 --data hHEP --net STRING
python main.py --num 1000 --data mDC --net STRING
python main.py --num 1000 --data mESC --net STRING
python main.py --num 1000 --data mHSC-E --net STRING
python main.py --num 1000 --data mHSC-GM --net STRING
python main.py --num 1000 --data mHSC-L --net STRING

## Non-specific
python main.py --num 1000 --data hESC --net Non-Specific
python main.py --num 1000 --data hHEP --net Non-Specific
python main.py --num 1000 --data mDC --net Non-Specific
python main.py --num 1000 --data mESC --net Non-Specific
python main.py --num 1000 --data mHSC-E --net Non-Specific
python main.py --num 1000 --data mHSC-GM --net Non-Specific
python main.py --num 1000 --data mHSC-L --net Non-Specific

## Lofgof
python main.py --num 1000 --data mESC --net Lofgof

## Specific
python main.py --num 1000 --data hESC --net Specific
python main.py --num 1000 --data hHEP --net Specific
python main.py --num 1000 --data mDC --net Specific
python main.py --num 1000 --data mESC --net Specific
python main.py --num 1000 --data mHSC-E --net Specific
python main.py --num 1000 --data mHSC-GM --net Specific
python main.py --num 1000 --data mHSC-L --net Specific

