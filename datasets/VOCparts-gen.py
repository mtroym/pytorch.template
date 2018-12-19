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


VALID_CLASS = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']




s1 = """    <object>
        <name>{0}</name>
        <pose>Unspecified</pose>
        <truncated>{1}</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{2}</xmin>
            <ymin>{3}</ymin>
            <xmax>{4}</xmax>
            <ymax>{5}</ymax>
        </bndbox>
    </object>"""
s2 = """<annotation>
    <folder>VOC2010</folder>
    <filename>{0}</filename>
    <source>
        <database>VOC2010</database>
        <annotation>VOC2010</annotation>
        <image>VOC2010</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>VOC2010</name>
    </owner>
    <size>
        <width>{1}</width>
        <height>{2}</height>
        <depth>{3}</depth>
    </size>
    <segmented>0</segmented>{4}
</annotation>
"""


def gen_xml(targets, imgPath, count):
    imgID = "0" * (6 - len(str(count))) + str(count)
    img = cv2.imread(imgPath)
    height, width, depth = img.shape
    obj = ""
    for i in range(len(targets)):
        cls = targets[i][0]
        x1,y1,x2,y2 = targets[i][1]
        if (y1 >= height or x1 >= width or x2 >= width or y2 >= height):
            print(imgID)
            print(x1,y1,x2,y2)
        obj += '\n' + s1.format(cls, 1, x1, y1, x2, y2)
    newImgPath = "/home/maoym/Develop/data/VOCparts/VOCParts2018/JPEGImage/" + imgID + ".jpg"
    cv2.imwrite(newImgPath, img)
    obj_final = s2.format(imgID +'.jpg', height, width, depth, obj)
    xmlPath = "/home/maoym/Develop/data/VOCparts/VOCParts2018/Annotations/" + imgID + ".xml"
    f = open(xmlPath, "w")
    f.write(obj_final)
    f.close()
    return imgID
        


def findBBox(mask):
    return cv2.boundingRect(mask)

def decodeBBox(matpath):
    data = sio.loadmat(matpath)
    total = []
    for info in (data['anno'][0][0]):
        for cls in info[0]:
            _class = cls[0][0]
            if _class in VALID_CLASS:
                parts = cls[3]
                if len(parts) == 0:
                    continue
                if (parts[0][0][0][0] == 'head'):
                    xywh = np.array(cv2.boundingRect(parts[0][0][1]))
                    total.append((_class, xywh_to_xyxy(xywh)))
    return total

def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    baseDir = os.path.join(opt.data, 'VOC2010', 'JPEGImages')
    print("=> Generating list of data")

    trainImgPath = []
    trainTarget = []
    trainClass = []

    print('=> decoding data set PART OF >HEADS<')
    count = 0
    trainval = []
    for photos in os.listdir(os.path.join(opt.data, 'VOC2010' , 'Annotations_Part')):
        imgID = photos.replace('.mat', '')
        imgPath = os.path.join(opt.data, 'VOC2010', 'JPEGImages', imgID + '.jpg')
        matPath = os.path.join(opt.data, 'VOC2010', 'Annotations_Part', photos)
        assert os.path.exists(imgPath) , 'Image not found: ' + imgID
        
        targets = decodeBBox(matPath)
        if len(targets) != 0:
            count += 1
            new_ID = gen_xml(targets, imgPath, count)
            trainTarget.append(targets)
            trainImgPath.append(imgPath)
            im_o = cv2.imread(imgPath)
            for targ in targets:
                cv2.rectangle(im_o, (targ[1][0],targ[1][1]), (targ[1][2],targ[1][3]), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(opt.data, 'VOC2010', 'JPEGImages_bbox', imgID + '_bbox.jpg'), im_o)
            print("=> done write," + new_ID)
            trainval.append(new_ID)
        
    numTrainImages = len(trainImgPath)
    print('#Training images: ' + str(numTrainImages))
    
    print("=> Shuffling")
    trainShuffle = torch.randperm(numTrainImages)
    trainImgPath = [trainImgPath[i] for i in trainShuffle]
    trainTarget = [trainTarget[i] for i in trainShuffle]
    
    trainval = [trainval[i] for i in trainShuffle]
    
    train_ratio = 0.9
    
    train = trainval[0:int(0.9*numTrainImages)]
    val = trainval[int(0.9*numTrainImages):]
    
    def save_split(img_list, split, path):
        save_path = path + split + ".txt"
        f = open(save_path, "w")
        f.write("\n".join(img_list))
        f.close()

    split_path = "/home/maoym/Develop/data/VOCparts/VOCParts2018/ImageSets/Main/"
    for split in ["trainval", "train", "val"]:
        save_split(eval(split), split, split_path)
        
    exit(0)
    info = {'basedir' : opt.data,
            'all' : {
                'imagePath'  : trainImgPath,
                'target'     : trainTarget,
                },
            }
    torch.save(info, cacheFile)
    return info
