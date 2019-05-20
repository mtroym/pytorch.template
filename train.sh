#!/usr/bin/env bash
python3 main.py --data /Users/tony/Develop/data/ \
--batchSize 2 --epochNum 0 \
--netType PSPNet --dataset IMaterialistFashion \
--metrics '[]' --LR 0.000001 \
--nThreads 0 --nEpochs 110 \
--logDir ../log/ --debug F \
--logNum 1 --optimizer SGD --epochNum -1
