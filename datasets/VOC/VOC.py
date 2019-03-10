from PIL import Image
import albumentations as albu
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import datasets.transforms as t
from datasets.VOC.decodeseg import encode_segmap
# https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/train.py

def bgr2rgb(im):
    b = im[..., 0]
    g = im[..., 1]
    r = im[..., 2]
    return np.stack([r, g, b], 2)


class VOC(Dataset):
    def __init__(self, imageInfo, opt, split):
        print("=> THe input CHANNELs are BGR not others.")
        self.inputSize = (375, 500)
        self.input_dim = 3
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.ignore_index = 255
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
        self.target_size = (self.inputSize[0] + 1, self.inputSize[1] + 1)
        if split == 'train':
            self.resizor = albu.Compose([albu.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),
                                         t.PadIfNeededRightBottom(min_height=self.target_size[0],
                                                                  min_width=self.target_size[1],
                                                                  value=0, ignore_index=self.ignore_index, p=1.0),
                                         albu.RandomCrop(height=self.target_size[0], width=self.target_size[1],
                                                         p=1.0)])
        else:
            self.resizor = albu.Compose([t.PadIfNeededRightBottom(min_height=self.target_size[0],
                                                                  min_width=self.target_size[1], value=0,
                                                                  ignore_index=self.ignore_index, p=1.0),
                                         albu.Crop(x_min=0, x_max=self.target_size[1], y_min=0,
                                                   y_max=self.target_size[0])])

    def __getitem__(self, index):
        path, target = self.imageInfo[index]
        image = np.array(Image.open(path))
        label = np.array(Image.open(target))


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
        #
        # if self.split == 'train':
        #     im = t.addNoise(im, 0.001, 0.001)
        # im, xml = t.randomSizeCrop(im, xml, 0.9)
        #  >>> how to resize....

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
    opt.inputSize = myInstance.inputSize
    opt.input_dim = myInstance.input_dim
    return myInstance


def main():
    img_gray = np.eye(100)
    img_gray[1:90, 10:20] = 2
    print(np.unique(img_gray))

    im = cv2.resize(img_gray, (50, 50), cv2.INTER_AREA)
    print(np.unique(im))
    print(im.shape)


if __name__ == '__main__':
    main()
