import os
import csv
import math
import torch
import numpy as np
import subprocess
import xml.dom.minidom


def decodeSol(xmlPath):
    dom = xml.dom.minidom.parse(xmlPath)
    objects = (dom.getElementsByTagName("object"))
    name = None
    maxAreas = []
    names = []
    results = []
    for obj in objects:
        name = str(obj.getElementsByTagName("name")[0].childNodes[0].data)
 
 
        nod = obj.getElementsByTagName("bndbox")[0]
        x_min = int(float(nod.getElementsByTagName("xmin")[0].childNodes[0].data))
        y_min = int(float(nod.getElementsByTagName("ymin")[0].childNodes[0].data))
        x_max = int(float(nod.getElementsByTagName("xmax")[0].childNodes[0].data))
        y_max = int(float(nod.getElementsByTagName("ymax")[0].childNodes[0].data))
        area = (y_max - y_min) * (x_max - x_min)
        if name in names:
            if area > maxAreas[names.index(name)]:
                results[names.index(name)] = [x_min, y_min, x_max, y_max]
                maxAreas[names.index(name)] = area
        else:
            names.append(name)
            results.append([x_min, y_min, x_max, y_max])
            maxAreas.append(area)
            
    return results, names

def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    baseDir = '/public/datasets/VOC/VOCdevkit/VOC2012/JPEGImages/'
    classDict = {}
    numclass = 0
    print("=> Generating list of data")

    VALNAME = []
    valImgPath = []
    valTarget = []
    valClass = []
    valVOCs = ['VOC2007']
    print('=> decoding validation set')
    for VOC in valVOCs:
        for photos in os.listdir(os.path.join(opt.data, 'val', VOC, 'Annotations')):
            imgID = photos.replace('.xml', '')
            VALNAME.append(imgID)
            target = os.path.join(opt.data, 'val', VOC, 'Annotations', imgID + '.xml')
            imgPath = os.path.join(opt.data, 'val', VOC, 'JPEGImages', imgID + '.jpg')
            windows, classNames = decodeSol(target)
            for obj in range(len(windows)):
                valImgPath.append(imgPath)
                valTarget.append(windows[obj])
                if (classNames[obj] not in classDict.keys()):
                    classDict[classNames[obj]] = numclass
                    numclass += 1
                valClass.append(classDict[classNames[obj]])            
            
            

    trainImgPath = []
    trainTarget = []
    trainClass = []
 
    trainVOCs = ['VOC2012', 'VOC2007']
    print('=> decoding training set')
    for VOC in trainVOCs:
        for photos in os.listdir(os.path.join(opt.data, 'train', VOC, 'Annotations')):
            imgID = photos.replace('.xml', '')
            if imgID.split('_')[0] == '2007' and (imgID.split('_')[1] in VALNAME):
                continue
            target = os.path.join(opt.data, 'train', VOC, 'Annotations', imgID + '.xml')
            imgPath = os.path.join(opt.data, 'train', VOC, 'JPEGImages', imgID + '.jpg')
            windows, classNames = decodeSol(target)
            assert len(windows) == len(classNames), "Error parsing target!"
            for obj in range(len(windows)):
                trainImgPath.append(imgPath)
                trainTarget.append(windows[obj])
                if (classNames[obj] not in classDict.keys()):
                    classDict[classNames[obj]] = numclass
                    numclass += 1
                trainClass.append(classDict[classNames[obj]])
 
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