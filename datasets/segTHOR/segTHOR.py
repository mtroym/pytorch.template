import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class SegTHOR(Dataset):
    def __init__(self, imageInfo, opt, split, return_h):
        self.return_h = return_h
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.pathData = imageInfo[split]
        self.inputSize = (252, 316)
        self.input_dim = 1
        self.boundary = [161, 413, 85, 401]  # (252, 316)
        self.mean, self.std = 0.456, 0.224
        opt.DSmode = 'file'
        self.DSmode = opt.DSmode
        self.img_slice = []
        self.GT_slice = []
        if self.DSmode == 'mem':
            last_patient_id = -1
            last_patient = None
            for (patient_id, sid), gtp, p in self.pathData:
                if patient_id != last_patient_id:
                    last_patient = nib.load(p).get_fdata(), nib.load(gtp).get_data()
                else:
                    pass
                img = last_patient[0][:, :, sid]  # image
                gt = last_patient[1][:, :, sid]  # ground truth
                self.img_slice.append(img)
                self.GT_slice.append(gt)
                last_patient_id = patient_id
        else:  # 'file'
            pass

    def __getitem__(self, index):
        # load file path, and load file.
        (pid, sid), gtp, p = self.pathData[index]
        img = nib.load(p).get_fdata()
        all_num = img.shape[-1]
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


def getInstance(info, opt, split):
    return_h = False
    if opt.netType == 'deeplabz':
        return_h = True
    myInstance = SegTHOR(info, opt, split, return_h)
    # print(myInstance.inputSize)
    opt.inputSize = myInstance.inputSize
    opt.input_dim = myInstance.input_dim
    return myInstance
