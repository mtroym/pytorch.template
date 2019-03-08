import cv2
import torch
import numpy as np
import datasets.transforms as t
from torch.utils.data import Dataset
from datasets.VOC.decodeseg import encode_segmap
from PIL import Image

def bgr2rgb(im):
    b = im[..., 0]
    g = im[..., 1]
    r = im[..., 2]
    return np.stack([r,g,b], 2)



class VOC(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.class_names = np.array([
            'background',
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'potted plant',
            'sheep',
            'sofa',
            'train',
            'tv/monitor',
        ])
        self.mean_bgr = np.array([0.485, 0.456, 0.406])
        self.var = np.array([0.229, 0.224, 0.225])

    def __getitem__(self, index):
        path, target = self.imageInfo[index]
        print(self.imageInfo[index])
        image = cv2.imread(path)
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float()
        image = np.asarray(image).astype(np.float)

        target = cv2.imread(target)
        target = bgr2rgb(target)
        target = encode_segmap(target)
        print(np.unique(target))
        target = torch.from_numpy(target)
        # target = np.asarray(target).astype(np.uint8)

        return image, target

 
    def __len__(self):
        return len(self.imageInfo)
 
    def preprocess(self, im, xml):
        if self.split == 'train':
            im = im

        # image -= self.mean_bgr
        # image /= self.var
        #

        im = np.asarray(im)
        im = t.normalize(im, self.mean_bgr, self.var)
        im = np.transpose(im, (2, 0, 1))
        return im, xml


def getInstance(info, opt, split):
    myInstance = VOC(info, opt, split)
    return myInstance
 