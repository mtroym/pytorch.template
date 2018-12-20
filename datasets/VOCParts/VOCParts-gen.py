import os
import csv
import math
import torch
import numpy as np
import subprocess
import xml.dom.minidom
import scipy.io as sio  
import cv2
from util.ds_utils import xywh_to_xyxy
import util.visualize as vis
from matplotlib import pyplot as plt

DATASET_NAME = "VOCParts"
DATASET_PATH = "VOCParts2018_v1"
TRAIN_ANNO_CSV = "train_annotations.csv"
VAL_ANNO_CSV = "val_annotations.csv"
CLASS_LIST = "class_list.csv"

def loadClasses(csvPath):
    with open(csvPath, 'r', newline='') as f:
        clsIdxs = f.readlines()
    result = {}
    for classID in clsIdxs:
        name, ID = classID[:-1].split(',')
        if name in result:
            raise ValueError('duplicate class name: \'{}\''.format(name))
        result[name] = int(ID)
    return result

def loadAnnos(line, classes):
    path, x1, x2, y1, y2, name = line.split(',')
    if (x1, y1, x2, y2, name) == ('', '', '', '', ''):
        return None
    fileName = path.split('/')[-1]
    annotation = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': name[:-1]}
    return path, annotation
    
def execSplit(csvPath, classes):
    f = open(csvPath, 'r', newline='')
    results = {}
    classCount = classes.copy()
    for i in classCount:
        classCount[i] = 0
    for pathAnnos in f.readlines():
        path, annos = loadAnnos(pathAnnos, classes)
        classCount[annos['class']] += 1
        if path not in results:
            results[path] = [annos]
        else:
            results[path].append(annos)
    print("COUNT: - " + csvPath + '\n\t\t' + str(classCount))
    return results

def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    baseDir = os.path.join(opt.data, DATASET_PATH, 'JPEGImages')
    print("=> Generating list of data")
    classesPath = os.path.join(opt.data, DATASET_PATH, CLASS_LIST)
    classes = loadClasses(classesPath)
    trainCSVPath = os.path.join(opt.data, DATASET_PATH, TRAIN_ANNO_CSV)
    train = execSplit(trainCSVPath, classes)
    valCSVPath = os.path.join(opt.data, DATASET_PATH, VAL_ANNO_CSV)
    val = execSplit(valCSVPath, classes)
    info = {
        'classDict' : classes,
        'basedir' : opt.data,
        'val' : val,
        'train': train
    }
#     print(info.keys())
    torch.save(info, cacheFilePath)
    return info
