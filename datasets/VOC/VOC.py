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
        print("=> THe input CHANNELs are BGR not others.")
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
        image = cv2.imread(path)
        image = np.asarray(image)

        target = cv2.imread(target)
        target = bgr2rgb(target)
        target = encode_segmap(target)

        image, target = self.preprocess(image, target)
        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target)

        return image, target

 
    def __len__(self):
        return len(self.imageInfo)
 
    def preprocess(self, im, xml):
        """
        :param im: np.array of size(h, w, c)
        :param xml: np.array of size(h, w)
        :return: same results.
        """
        im = im.astype(float)
        im = t.scaleRGB(im)
        im -= self.mean_bgr
        im /= self.var

        if self.split == 'train':
            im = t.addNoise(im, 0.001, 0.001)
            im, xml = t.randomSizeCrop(im, xml, 0.9)

        im = im.resize(new_shape=(375, 500))
        xml = xml.resize(new_shape=(375, 500))
        im = np.transpose(im, (2, 0, 1))
        return im, xml


    def postprocess(self, im, xml):
        """
        :param im: np.array of size(c, h, w)
        :param xml: np.array of size(h, w)
        :return: same results.
        """
        im = im.transpose((1, 2, 0))
        im = t.unScaleRGB(im)
        im *= self.var
        im += self.mean_bgr
        # TODO add masks...
        return im, xml



def getInstance(info, opt, split):
    myInstance = VOC(info, opt, split)
    return myInstance
 