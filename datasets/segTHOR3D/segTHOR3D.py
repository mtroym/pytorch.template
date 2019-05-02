import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class SegTHORz(Dataset):
    def __init__(self, imageInfo, opt, split, return_h):
        self.return_h = return_h
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.pathData = imageInfo[split][2]
        self.inputSize = (252, 316)
        self.input_dim = 1
        self.boundary = [161, 413, 85, 401]  # (252, 316)
        self.mean, self.std = 0.456, 0.224
        opt.DSmode = 'file'
        self.DSmode = opt.DSmode
        self.img_slice = []
        self.GT_slice = []

    def __getitem__(self, index):
        # load file path, and load file.
        (pid, sid), gtp, p = self.pathData[index]
        img = nib.load(p).get_fdata()
        all_num = img.shape[2]
        img = img[:, :, sid]
        gt = nib.load(gtp).get_data()[:, :, sid]

        # some preprocessing...
        img, gt = self._transform(img, gt)
        img_ = np.array(img)
        image = torch.from_numpy(img_).float()
        target = torch.from_numpy(gt)
        h = (sid*1.0)/all_num - 0.5
        h = np.ones(1) * h
        h = torch.from_numpy(h[np.newaxis, np.newaxis, ...]).float()
        # convert to tensor.
        if self.return_h:
            return (pid, sid), image[np.newaxis, ...], target, h
        else:
            return (pid, sid), image[np.newaxis, ...], target

    def __len__(self):
        return len(self.pathData)

    def _transform(self, img, mask, low_range=-200, high_range=200, ):
        # thershold [-200, 200] -> normalize -> crop
        _img = img.copy()
        _img[img > high_range] = high_range
        _img[img < low_range] = low_range

        _img /= 255
        _img -= self.mean
        _img /= self.std

        top, bottom, left, right = self.boundary
        return _img[top:bottom, left:right], mask[top:bottom, left:right]

class SegTHORy(Dataset):
    def __init__(self, imageInfo, opt, split, return_h):
        self.return_h = return_h
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.pathData = imageInfo[split][1]
        self.inputSize = (252, 180)
        self.input_dim = 1
        self.boundary = [161, 413, 85, 401]  # (252, 316)
        self.mean, self.std = 0.456, 0.224
        self.img_slice = []
        self.GT_slice = []

    def __getitem__(self, index):
        # load file path, and load file.
        (pid, sid), gtp, p = self.pathData[index]
        img = nib.load(p).get_fdata()
        all_num = img.shape[1]
        img = Image.fromarray(img[:, sid, :])
        gt = Image.fromarray(nib.load(gtp).get_data()[:, sid, :])

        img.resize((512, 180))
        gt.resize((512, 180))

        img = np.array(img)
        gt = np.array(gt)

        # some preprocessing...
        img, gt = self._transform(img, gt)
        img_ = np.array(img)
        image = torch.from_numpy(img_).float()
        target = torch.from_numpy(gt)
        h = (sid*1.0)/all_num - 0.5
        h = np.ones(1) * h
        h = torch.from_numpy(h[np.newaxis, np.newaxis, ...]).float()
        # convert to tensor.
        if self.return_h:
            return (pid, sid), image[np.newaxis, ...], target, h
        else:
            return (pid, sid), image[np.newaxis, ...], target

    def __len__(self):
        return len(self.pathData)

    def _transform(self, img, mask, low_range=-200, high_range=200, ):
        # thershold [-200, 200] -> normalize -> crop
        _img = img.copy()
        _img[img > high_range] = high_range
        _img[img < low_range] = low_range

        _img /= 255
        _img -= self.mean
        _img /= self.std

        top, bottom, left, right = self.boundary
        return _img[top:bottom, :], mask[top:bottom, :]

class SegTHORx(Dataset):
    def __init__(self, imageInfo, opt, split, return_h):
        self.return_h = return_h
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.pathData = imageInfo[split][0]
        self.inputSize = (316, 180)
        self.input_dim = 1
        self.boundary = [161, 413, 85, 401]  # (252, 316)
        self.mean, self.std = 0.456, 0.224
        opt.DSmode = 'file'
        self.DSmode = opt.DSmode
        self.img_slice = []
        self.GT_slice = []

    def __getitem__(self, index):
        # load file path, and load file.
        (pid, sid), gtp, p = self.pathData[index]
        img = nib.load(p).get_fdata()
        all_num = img.shape[0]
        img = Image.fromarray(img[sid, :, :])
        gt = Image.fromarray(nib.load(gtp).get_data()[sid, :, :])

        img = img.resize((512, 180))
        gt = gt.resize((512, 180))

        img = np.array(img)
        gt = np.array(gt).dtype(int)

        # some preprocessing...
        img, gt = self._transform(img, gt)
        img_ = np.array(img)
        image = torch.from_numpy(img_).float()
        target = torch.from_numpy(gt)
        h = (sid*1.0)/all_num - 0.5
        h = np.ones(1) * h
        h = torch.from_numpy(h[np.newaxis, np.newaxis, ...]).float()
        # convert to tensor.
        if self.return_h:
            return (pid, sid), image[np.newaxis, ...], target, h
        else:
            return (pid, sid), image[np.newaxis, ...], target

    def __len__(self):
        return len(self.pathData)

    def _transform(self, img, mask, low_range=-200, high_range=200, ):
        # thershold [-200, 200] -> normalize -> crop
        _img = img.copy()
        _img[img > high_range] = high_range
        _img[img < low_range] = low_range

        _img /= 255
        _img -= self.mean
        _img /= self.std

        top, bottom, left, right = self.boundary
        return _img[left:right,  :], mask[left:right, :]



def getInstance(info, opt, split):
    get_h = False
    if opt.netType == 'deeplabz' or opt.netType == 'deeplabz3d':
        get_h = True
    myInstancex = SegTHORx(info, opt, split, get_h)
    myInstancey = SegTHORy(info, opt, split, get_h)
    myInstancez = SegTHORz(info, opt, split, get_h)
    # print(myInstance.inputSize)
    opt.inputSize = [myInstancex.inputSize, myInstancey.inputSize, myInstancez.inputSize]
    opt.input_dim = 1
    return myInstancex, myInstancey, myInstancez
