#!/usr/bin/env bash
python3 main.py --data ../data/ --batchSize 10 --epochNum 0  --netType PSPNet  --dataset  IMaterialistFashion --metrics '[]' --LR 0.000001 --nThreads 0 --debug T --logNum 1  --optimizer SGD --epochNum -1
