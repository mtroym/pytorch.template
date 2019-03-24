import copy
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from albumentations import Compose, IAAPiecewiseAffine, ElasticTransform
from torch.utils.data import Dataset

debug = 0


class DAVIS(Dataset):
    def __init__(self, info, opt, split):
        self.input_size = (854, 480)
        self.instance_all = info[split]
        self.opt = opt
        self.input_channel = 4

    def __len__(self):
        return len(self.instance_all)

    def __getitem__(self, i):
        image_path, gt_path, instance_num = self.instance_all[i]
        image = Image.open(image_path)
        ground_truth = Image.open(gt_path)

        ground_truth.load()
        image.load()

        image.resize(self.input_size)
        ground_truth.resize(self.input_size)

        image = np.array(image, dtype=np.uint8)
        ground_truth = np.array(ground_truth, dtype=np.uint8)

        label = np.zeros_like(ground_truth)
        label[ground_truth == instance_num] = 1

        # Augmentation for mask input; single instance.
        mask = copy.deepcopy(label)
        mask = self.mask_transformation(mask)
        if debug:
            mask_ = np.stack((mask, mask, mask), -1)
            img_ = (1 - mask_.astype(np.float)) * image.astype(np.float) + mask_.astype(np.float) * (
                    np.array([255.0, 0, 0]) * 0.3 + 0.7 * image.astype(np.float))
            plt.imshow(img_.astype(np.int))
            plt.show()
        mask = mask * 400 - 200  # scale 0 ~ 1 to -200 ~ 200
        image_mask = np.concatenate((image, mask[..., np.newaxis]), 2)
        # adjust the dimension for the binary crossentropy loss.
        return image_mask, label

    @staticmethod
    def mask_transformation(mask):
        """
        To simulate the deformation noise. affine transform + non-rigid deformations(TPS) + dilation
        This is from paper: `during offline training we generate input masks by deforming the
        annotated masks via affine transformation as well as non-rigid
        deformations via thin-plate splines [4], followed by a coarsening
        step (dilation morphological operation) to remove details of the object contour. `
        :param mask: np.array of shape (h, w)
        :return: np.array of shape (h, w) after deformation noise.
        """
        aug = Compose([
            # TODO: add another mask transformation ?
            IAAPiecewiseAffine(p=0.4),  # affine + non-rigid deformation
            ElasticTransform(p=0.6, alpha=50, sigma=50, alpha_affine=20,
                             border_mode=cv2.BORDER_CONSTANT, always_apply=True),
        ])
        augmented = aug(image=mask)
        mask = augmented['image']
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=random.randrange(1, 10))
        return mask


def getInstance(info, opt, split):
    myInstance = DAVIS(info, opt, split)
    opt.inputSize = myInstance.input_size
    opt.input_channel = myInstance.input_channel
    return myInstance
