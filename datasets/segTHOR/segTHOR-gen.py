import os

import nibabel as nib
import numpy as np
import torch


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限
    training_data_path = "/Users/tony/Downloads/train/"
    split_ratio = 0.5
    patients = os.listdir(training_data_path)
    patients.remove('.DS_Store')
    train_pa = patients[:int(len(patients) * split_ratio)]
    val_pa = patients[int(len(patients) * split_ratio):]
    print(">>>=====> Generating list of dat ....")
    info = {
        'basedir': training_data_path,
        'val': loadPaths(training_data_path, val_pa),
        'train': loadPaths(training_data_path, train_pa)
    }
    torch.save(info, cacheFilePath)
    return info


def loadPaths(base, pa):
    X = []
    Y = []
    preserving_ratio = 0.001
    count = 0
    for patients in pa:
        print(patients)
        # file contains GT.nii // patient_xx.nii.gz
        GT_path = os.path.join(base, patients, 'GT.nii.gz')
        img_path = os.path.join(base, patients, patients + '.nii.gz')
        # print(GT_path, img_path)
        GT = nib.load(GT_path).get_data()
        img = nib.load(img_path).get_data()
        # print(img.shape, GT.shape)
        # print(np.unique(GT))
        img_3d_max = np.amax(img)
        img = img / img_3d_max * 255
        for i in range(img.shape[2]):
            GT_2d = GT[:, :, i]
            if float(np.count_nonzero(GT_2d)) / GT_2d.size < preserving_ratio:
                continue
            img_2d = img[:, :, i]
            img_2d = img_2d / 127.5 - 1
            X.append(img_2d)
            Y.append(GT_2d)
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.uint8)
    print(X.shape, Y.shape)
    # return path to load.
    return ""
