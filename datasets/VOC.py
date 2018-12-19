import cv2
import torch
import numpy as np
import datasets.transforms as t
from torch.utils.data import Dataset

class VOC(Dataset):
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
        clas = self.imageInfo['class'][index]
        classVec = np.zeros((self.numClass,1))
        classVec[clas] = 1
 
 
        w, h, _ = image.shape
        image = self.preprocess(image)
        # image = torch.from_numpy(image).float()
 
        image = [torch.from_numpy(image).float(), torch.from_numpy(classVec).float()]
 
        target = self.imageInfo['target'][index]
        target = np.array(target)
        target = self.preprocessTarget(target, w, h)
        target = torch.from_numpy(target).float()
 
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
    
    

    def preprocessTarget(self, t, w, h):
        x_min = t[0] / float(h)
        y_min = t[1] / float(w)
        x_max = t[2] / float(h)
        y_max = t[3] / float(w)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        x_delta = np.log(x_max - x_min)
        y_delta = np.log(y_max - y_min)
        return np.array([x_mid, y_mid, x_delta, y_delta])
    
    def postprocess(self):
        def process(im):
            mean = torch.Tensor([0.485, 0.456, 0.406])
            std = torch.Tensor([0.229, 0.224, 0.225])
            im = np.transpose(im, (1, 2, 0))
            im = t.unNormalize(im, mean, std)
            return im
        return process
 
    def postprocessTarget(self):
        def process(t):
            x_mid = t[0] * self.opt.imgDim
            y_mid = t[1] * self.opt.imgDim
            x_delta = np.exp(t[2]) * self.opt.imgDim
            y_delta = np.exp(t[3]) * self.opt.imgDim
            x_min = x_mid - x_delta / 2
            y_min = y_mid - y_delta / 2
            x_max = x_mid + x_delta / 2
            y_max = y_mid + y_delta / 2
            return np.array([x_min, y_min, x_max, y_max])
        return process
 
    def postprocessHeat(self):
        def process(im):
            im = np.transpose(im, (1, 2, 0))
            im *= 255
            return im
        return process
    

def getInstance(info, opt, split):
    myInstance = VOC(info, opt, split)
    return myInstance
 