import os
import csv
import math
import torch
import numpy as np
import subprocess
from pathlib import Path

def exec(opt, cacheFile):
    datasetRoot = Path('/public/datasets/imagenet/')
    
    classDict = {}
    numClass = 0
    
    trainPath = datasetRoot / 'train'
    valPath = datasetRoot / 'val'
    
    
    print("=> Generating list of data")
    
    trainImgPath = []
    trainTarget = []


    
    for imgPath in trainPath.iterdir():
        className = imgPath.parts[-1]
        if (className not in classDict.keys()):
            classDict[className] = numClass
            numClass += 1
        target = classDict[className]
        for p in imgPath.iterdir():
            trainImgPath.append(str(p))
            trainTarget.append(target)
            
    print("=> Training data already setup!")
    
    valImgPath = []
    valTarget = []   
    for imgPath in valPath.iterdir():
        className = imgPath.parts[-1]
        target = classDict[className]
        for p in imgPath.iterdir():
            valImgPath.append(str(p))
            valTarget.append(target)
    
    print("=> Validation already setup!")

    
    numTrainImages = len(trainImgPath)
    numValImages = len(valImgPath)
    
    print('#Training images: ' + str(numTrainImages))
    print('#Testing images: ' + str(numValImages))

    print("=> Shuffling")
    trainShuffle = torch.randperm(numTrainImages)
    valShuffle = torch.randperm(numValImages)
    trainImgPath = [trainImgPath[i] for i in trainShuffle]
    trainTarget = [trainTarget[i] for i in trainShuffle]
    valImgPath = [valImgPath[i] for i in valShuffle]
    valTarget = [valTarget[i] for i in valShuffle]
    
    info = {'basedir' : opt.data,
            'classDict' : classDict,
            'train' : {
                'imagePath'  : trainImgPath,
                'target'     : trainTarget,
                },
            'val' : {
                'imagePath'  : valImgPath,
                'target'     : valTarget,
                }
            }

    torch.save(info, cacheFile)
    return info
