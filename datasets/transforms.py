import math
import random

import cv2
import numpy as np
import torch
from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform, DualTransform


# ipt is nparray with dimension (height, width, channel)
# xml is nparray with dimension (height, width)

def addNoise(ipt, miu, std):
    noise = np.random.normal(miu, std, ipt.shape)
    noise = np.float32(noise)
    return ipt + noise


def thAddNoise(ipt, miu, std):
    noise = np.random.normal(miu, std, ipt.size())
    noise = torch.from_numpy(np.float32(noise))
    return ipt + noise


def scaleRGB(ipt):
    return np.float32(ipt / 255)


def unScaleRGB(ipt):
    opt = ipt * 255
    opt = opt.astype(np.uint8)
    return opt


def normalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] - mean[0]) / std[0]
    ipt[:][:][1] = (ipt[:][:][1] - mean[1]) / std[1]
    ipt[:][:][2] = (ipt[:][:][2] - mean[2]) / std[2]
    return ipt


def unNormalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] * std[0]) + mean[0]
    ipt[:][:][1] = (ipt[:][:][1] * std[1]) + mean[1]
    ipt[:][:][2] = (ipt[:][:][2] * std[2]) + mean[2]
    return ipt


def randomFlip(ipt, xml):
    if random.uniform(0, 1) > 0.5:
        ipt = np.fliplr(ipt).copy()
        xml = np.fliplr(xml).copy()
    return ipt, xml


def randomCrop(ipt, xml, size):
    origH = ipt.shape[0]
    origW = ipt.shape[1]
    newH = size[0]
    newW = size[1]
    startH = random.randint(0, origH - newH)
    startW = random.randint(0, origW - newW)
    ipt = ipt[startH: startH + newH, startW: startW + newW, :]
    xml = xml[startH: startH + newH, startW: startW + newW]
    return ipt, xml


def randomSizeCrop(ipt, xml, LowBound):
    newH = math.floor(random.uniform(LowBound, 1) * ipt.shape[0])
    while newH % 8 != 0:
        newH -= 1
    newW = math.floor(random.uniform(LowBound, 1) * ipt.shape[1])
    while newW % 8 != 0:
        newW -= 1
    return randomCrop(ipt, xml, (newH, newW))


# ==================

def apply_motion_blur(image, count):
    """
    https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    image_t = image.copy()
    imshape = image_t.shape
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    i = imshape[1] * 3 // 4 - 10 * count
    while i <= imshape[1]:
        image_t[:, i:, :] = cv2.filter2D(image_t[:, i:, :], -1, kernel_motion_blur)
        image_t[:, :imshape[1] - i, :] = cv2.filter2D(image_t[:, :imshape[1] - i, :], -1, kernel_motion_blur)
        i += imshape[1] // 25 - count
        count += 1
    color_image = image_t
    return color_image


def rotate(img, angle, interpolation, border_mode, border_value=None):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height),
                         flags=interpolation, borderMode=border_mode, borderValue=border_value)
    return img


class AddSpeed(ImageOnlyTransform):
    def __init__(self, speed_coef=-1, p=.5):
        super(AddSpeed).__init__(p)
        assert speed_coef == -1 or 0 <= speed_coef <= 1
        self.speed_coef = speed_coef

    def apply(self, img, count=7, **params):
        return apply_motion_blur(img, count)

    def get_params(self):
        if self.speed_coef == -1:
            return {'count': int(15 * random.uniform(0, 1))}
        else:
            return {'count': int(15 * self.speed_coef)}


class Rotate(DualTransform):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, border_value=255, always_apply=False, p=.5):
        super(Rotate).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value

    def apply(self, img, angle=0, **params):
        return rotate(img, angle, interpolation=self.interpolation, border_mode=self.border_mode)

    def apply_to_mask(self, img, angle=0, **params):
        return rotate(img, angle, interpolation=cv2.INTER_NEAREST,
                      border_mode=cv2.BORDER_CONSTANT, border_value=self.border_value)

    def get_params(self):
        return {'angle': random.uniform(self.limit[0], self.limit[1])}


class PadIfNeededRightBottom(DualTransform):
    def __init__(self, min_height=769, min_width=769, border_mode=cv2.BORDER_CONSTANT,
                 value=0, ignore_index=255, always_apply=False, p=1.0):
        super(PadIfNeededRightBottom).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.ignore_index = ignore_index

    def apply(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height - img_height)
        pad_width = max(0, self.min_width - img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=self.value)

    def apply_to_mask(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height - img_height)
        pad_width = max(0, self.min_width - img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width)), 'constant', constant_values=self.ignore_index)
