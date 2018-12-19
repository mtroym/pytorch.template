import os
from pathlib import Path
import numpy as np

total = []

trainDir = '/Users/yifan/Downloads/imagenet20/train'
valDir = '/Users/yifan/Downloads/imagenet20/val'

for i in os.listdir(valDir):
    trainClass = os.path.join(trainDir, i)
    i = os.path.join(valDir, i)
    if not os.path.exists(trainClass):
        os.makedirs(trainClass)
    # if os.path.isdir(i):
    #     print(i)
    #     print(trainClass)
    #     totalClass = [name for name in os.listdir(i)]
    #     numClass = len(totalClass)
    #     count = 0
    #     while count < 0.95 * numClass:
    #         imgName = os.path.join(i, totalClass[count])
    #         imgDes = os.path.join(trainClass, totalClass[count])
    #         os.rename(imgName, imgDes)
    #         count += 1
    #     total.append(numClass)
    #     print(numClass)
# total = np.array(total)
# print(np.min(total), np.max(total), np.average(total), np.std(total))

for i in os.listdir(valDir):
    trainClass = os.path.join(trainDir, i)
    valClass = os.path.join(valDir, i)
    if os.path.isdir(trainClass):
        print(trainClass)
        print(valClass)
        numTrain = len([name for name in os.listdir(trainClass)])
        numVal = len([name for name in os.listdir(valClass)])
        print(numTrain, numVal)
