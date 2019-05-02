python3 main.py --data /root/data \
--batchSize 20 --epochNum -1 \
--netType deeplab --dataset segTHOR \
--metrics '[Dice]' --LR 0.0005 \
--nThreads 10 --nEpochs 100 \
--logDir ../log/ --debug F \
--logNum 1 --optimizer SGD \
--suffix BS20 \
--backbone Resnet \
--DSmode file --epochNum -1
