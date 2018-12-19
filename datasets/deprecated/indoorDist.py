from torch.utils.data.dataset import *
import torch
import os
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
        image = np.asarray(image)

        gtPath = path.replace('_rgb.jpg', '_gt.png')
        gtImage = cv2.imread(gtPath)
        gtImage = np.asarray(gtImage)

        image = self.preprocess(image)
        image = self.swapaxes(image)
        gtImage = self.swapaxes(gtImage)

        tmp = np.zeros((1, self.opt.imgDim, self.opt.imgDim))
        tmp[0] = gtImage[0]
        gtImage = tmp

        image = torch.from_numpy(image).float()
        gtImage = torch.from_numpy(gtImage).float()

        return image, gtImage

    def __len__(self):
        return len(self.imageInfo['imagePath'])

    def swapaxes(self, im):
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
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
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            im = t.unNormalize(im, mean, std)
            return im
        return process

    def postprocessGT(self):
        def process(im):
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            return im
        return process

def getInstance(info, opt, split):
    myInstance = myDataset(info, opt, split)
    return myInstance
