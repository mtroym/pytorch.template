python3 main.py --data /root/data \
--batchSize 6 --epochNum 0 \
--netType deeplab --dataset segTHOR \
--metrics '[Dice]' --LR 0.0005 \
--nThreads 30 --nEpochs 100 \
--logDir ../log/ --debug F \
--logNum 1 --optimizer SGD \
--suffix BS24 \
--backbone Xception \
--DSmode file --epochNum -1
