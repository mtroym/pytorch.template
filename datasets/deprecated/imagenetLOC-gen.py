import os
import csv
import math
import torch
import numpy as np
import subprocess

def decodeSol(t):
    t = t.split()
    maxArea = 0
    i = 0
    name = ""
    while i in range(len(t)):
        if t[i][0] == 'n':
            x_min = int(t[i + 1])
            y_min = int(t[i + 2])
            x_max = int(t[i + 3])
            y_max = int(t[i + 4])
            area = (y_max - y_min) * (x_max - x_min)
            if area > maxArea:
                result = [x_min, y_min, x_max, y_max]
                maxArea = area
                name = t[i]
        i += 5
    return name, result

def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data

    baseDir = '/public/datasets/imagenet/'
    classDict = {}
    numclass = 0
    
    print("=> Generating list of data")
    
    trainImgPath = []
    trainTarget = []
    trainClass = []

    with open(os.path.join(opt.data, 'LOC_train_solution.csv')) as f:
        pathReader = csv.DictReader(f, dialect=csv.excel)
        for p in pathReader:
            imgID = p['ImageId']
            target = p['PredictionString']
            className, target = decodeSol(target)
            imgPath = os.path.join(baseDir, 'train', className, imgID + '.JPEG')
            trainImgPath.append(imgPath)
            trainTarget.append(target)
            if (className not in classDict.keys()):
                classDict[className] = numclass
                numclass += 1
            trainClass.append(classDict[className])
            
            
    valImgPath = []
    valTarget = []
    valClass = []    
    with open(os.path.join(opt.data, 'LOC_val_solution.csv')) as f:
        pathReader = csv.DictReader(f, dialect=csv.excel)
        for p in pathReader:
            imgID = p['ImageId']
            target = p['PredictionString']
            className, target = decodeSol(target)
            imgPath = os.path.join(baseDir, 'val', className, imgID + '.JPEG')
            valImgPath.append(imgPath)
            valTarget.append(target)
            if (className not in classDict.keys()):
                classDict[className] = numclass
                numclass += 1
            valClass.append(classDict[className])
            
            
    numTrainImages = len(trainImgPath)
    numValImages = len(valImgPath)
    print('#Training images: ' + str(numTrainImages))
    print('#Testing images: ' + str(numValImages))

    print("=> Shuffling")
    trainShuffle = torch.randperm(numTrainImages)
    valShuffle = torch.randperm(numValImages)
    trainImgPath = [trainImgPath[i] for i in trainShuffle]
    trainTarget = [trainTarget[i] for i in trainShuffle]
    trainClass = [trainClass[i] for i in trainShuffle]
    valImgPath = [valImgPath[i] for i in valShuffle]
    valTarget = [valTarget[i] for i in valShuffle]
    valClass = [valClass[i] for i in valShuffle]
    
    info = {'basedir' : opt.data,
            'classDict' : classDict,
            'train' : {
                'imagePath'  : trainImgPath,
                'class'      : trainClass,
                'target'     : trainTarget,
                },
            'val' : {
                'imagePath'  : valImgPath,
                'class'      : valClass,
                'target'     : valTarget,
                }
            }

    torch.save(info, cacheFile)
    return info
