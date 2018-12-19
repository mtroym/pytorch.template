from torch.utils.data.dataset import *
import os
import torch
import numpy as np
import cv2
import datasets.transforms as t

class myDataset(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']

    def __getitem__(self, index):
        path = self.imageInfo['imagePath'][index]
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))
        image = np.asarray(image)

        image = self.preprocess(image)
        image = self.swapaxes(image)
        image = torch.from_numpy(image).float()

        return image, 0

    def __len__(self):
        return len(self.imageInfo['imagePath'])

    def swapaxes(self, im):
        im = np.transpose(im, (2, 0, 1))
        return im

    def preprocess(self, im):
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        im = t.normalize(im, mean, std)
        return im

    def postprocess(self):
        def process(im):
            mean = torch.Tensor([0.485, 0.456, 0.406])
            std = torch.Tensor([0.229, 0.224, 0.225])
            im = np.transpose(im, (1, 2, 0))
            im = t.unNormalize(im, mean, std)
            return im
        return process

def getInstance(info, opt, split):
    myInstance = myDataset(info, opt, split)
    return myInstance
