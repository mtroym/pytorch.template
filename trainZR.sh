python3 main.py --data /root/data \
--batchSize 20 --epochNum -1 \
--netType deeplabz --dataset segTHOR \
--metrics '[Dice]' --LR 0.0005 \
--nThreads 30 --nEpochs 200 \
--logDir ../log/ --debug F \
--logNum 1 --optimizer SGD \
--suffix BS24 \
--backbone Resnet \
--DSmode file --epochNum -1


