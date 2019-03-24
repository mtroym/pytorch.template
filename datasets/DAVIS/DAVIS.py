from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as albu

class DAVIS(Dataset):
    def __init__(self, info, opt, split):
        self.input_size = (480, 256)
        self.instance_all = info[split]

    def __len__(self):
        return len(self.instance_all)

    def __getitem__(self, i):
        image_path, gt_path, instance_num = self.instance_all[i]
        image = Image.open(image_path)
        ground_truth = np.array(Image.open(gt_path))
        label = np.zeros_like(ground_truth)
        label[ground_truth == instance_num] = 1

        # TODO; label distortion.

        return image, label




def getInstance(info, opt, split):
    myInstance = DAVIS(info, opt, split)
    # print(myInstance.inputSize)
    opt.inputSize = myInstance.input_size
    return myInstance
