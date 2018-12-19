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
    inputPath = findImages(opt, 'val')
    numImages = len(inputPath)
    numTrainImages = math.floor(numImages * opt.trainPctg)
    print('#Total images: ' + str(numImages))
    print('#Training images: ' + str(numTrainImages))

    print("=> Shuffling")
    shuffle = torch.randperm(numImages)
    trainShuffle = shuffle[:numTrainImages]
    valShuffle = shuffle[numTrainImages:]
    trainImgPath = [inputPath[i] for i in trainShuffle]
    valImgPath = inputPath

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
