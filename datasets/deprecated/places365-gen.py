import torch
import os
import math
import numpy as np
import subprocess
import torchvision

def findImages(opt, split):
    inputPath = subprocess.run(["find", os.path.join(opt.data, split), '-iname', '*.jpg'], check=True, stdout=subprocess.PIPE).stdout.decode("utf-8")
    inputPath = inputPath.split()
    inputPath.sort()

    return inputPath

def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data

    print("=> Generating list of data")
    trainPath = findImages(opt, 'train')
    valPath = findImages(opt, 'val')

    numTrain = len(trainPath)
    numVal = len(valPath)
    numImages = numTrain + numVal
    print('#Total images: ' + str(numImages))
    print('#Training images: ' + str(numTrain))

    print("=> Shuffling")
    trainShuffle = torch.randperm(numTrain)
    valShuffle = torch.randperm(numVal)

    trainImgPath = [trainPath[i] for i in trainShuffle]
    valImgPath = [valPath[i] for i in valShuffle]

    info = {'basedir' : opt.data,
            'train' : {
                'imagePath'  : trainImgPath,
                },
            'val' : {
                'imagePath'  : valImgPath,
                }
            }

    torch.save(info, cacheFile)
    return info
