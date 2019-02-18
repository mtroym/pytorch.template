import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import datasets.transforms as t


class DAVIS(Dataset):
    def __init__(self, info, opt, split):
        self.input_size = (122, 122)
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        raise NotImplementedError()

def getInstance(info, opt, split):
    myInstance = DAVIS(info, opt, split)
    # print(myInstance.inputSize)
    opt.inputSize = myInstance.inputSize
    return myInstance
