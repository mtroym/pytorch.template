import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class SegTHOR(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.pathData = imageInfo[split]
        self.inputSize = (252, 316)
        self.input_dim = 3
        self.boundary = [161, 413, 85, 401]  # (252, 316)
        self.mean, self.std = 0.456, 0.224

    def __getitem__(self, index):
        (pid, sid), gtp, p = self.pathData[index]
        img = nib.load(p).get_fdata()[:, :, sid]
        GT = nib.load(gtp).get_data()[:, :, sid]
        img, GT = self._transform(img, GT)
        img_ = np.array([img, img, img])
        image = torch.from_numpy(img_).float()
        target = torch.from_numpy(GT)
        return image, target

    def __len__(self):
        return len(self.pathData)

    def _transform(self, img, mask, low_range=-200, high_range=200, ):
        # thershold [-200, 200] -> normalize -> crop
        _img = img.copy()
        _img[img > high_range] = high_range
        _img[img < low_range] = low_range

        _img /= 255.0
        _img -= self.mean
        _img /= self.std

        top, bottom, left, right = self.boundary
        return _img[top:bottom, left:right], mask[top:bottom, left:right]


def getInstance(info, opt, split):
    myInstance = SegTHOR(info, opt, split)
    # print(myInstance.inputSize)
    opt.inputSize = myInstance.inputSize
    opt.input_dim = myInstance.input_dim
    return myInstance
