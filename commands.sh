#!/bin/bash

python train.py --stage=0 &&
python extract_alignments.py &&
python train.py --stage=1 &&
python train.py --stage=2 &&
python extract_alignments.py &&
python train.py --stage=3
