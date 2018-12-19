import cv2
import torch
import numpy as np
import datasets.transforms as t
from torch.utils.data import Dataset

class imagenet(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.classDict = imageInfo['classDict']
        self.numClass = len(imageInfo['classDict'])

    def __getitem__(self, index):
        path = self.imageInfo['imagePath'][index]

        image = cv2.imread(path)

        w, h, _ = image.shape
        image = self.preprocess(image)

        image = torch.from_numpy(image).float()

        target = self.imageInfo['target'][index]
        return image, target
    

    def __len__(self):
        return len(self.imageInfo['imagePath'])

    def preprocess(self, im):
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        im = cv2.resize(im, (self.opt.imgDim, self.opt.imgDim))
        im = np.asarray(im)
        im = t.normalize(im, mean, std)
        im = np.transpose(im, (2, 0, 1))
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
    myInstance = imagenet(info, opt, split)
    return myInstance
