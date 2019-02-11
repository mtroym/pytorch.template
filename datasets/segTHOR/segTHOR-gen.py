import os

import nibabel as nib
import numpy as np
import torch

from util.progbar import progbar

def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限
    training_data_path = os.path.join(opt.data, 'train')
    split_ratio = 0.5
    patients = [path for path in os.listdir(training_data_path) if path.startswith('Patient')]
    train_pa = patients[:int(len(patients) * split_ratio)]
    val_pa = patients[int(len(patients) * split_ratio):]
    print("************* Generating list of dat ....**************")
    info = {
        'basedir': training_data_path,
        'val': loadPaths(opt, training_data_path, val_pa, 'val'),
        'train': loadPaths(opt, training_data_path, train_pa, 'train')
    }
    print("************* Saved the cached file in {} **************".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info


def loadPaths(opt, base, pa, split):
    print('\t-> processing {} data'.format(split))
    X = []
    if not os.path.exists(os.path.join(base, split)):
        os.makedirs(os.path.join(base, split))
    preserving_ratio = 0.001
    bar = progbar(len(pa), width=opt.barwidth)
    for idx, patients in enumerate(pa):
        # file contains GT.nii // patient_xx.nii.gz
        GT_path = os.path.join(base, patients, 'GT.nii.gz')
        img_path = os.path.join(base, patients, patients + '.nii.gz')
        GT = nib.load(GT_path).get_data()
        img = nib.load(img_path).get_data()
        img_3d_max = np.amax(img)
        img = img / img_3d_max * 255
        for i in range(img.shape[2]):
            GT_2d = GT[:, :, i]
            if float(np.count_nonzero(GT_2d)) / GT_2d.size < preserving_ratio and split == 'train':
                continue
            img_2d = img[:, :, i]
            img_2d = img_2d / 127.5 - 1

            GT_2d = GT_2d.reshape((1, *GT_2d.shape))
            img_2d = img_2d.reshape((1, *img_2d.shape))
            pathALL = os.path.join(base, split, patients + '_{}'.format(i) + '.npy')
            np.save(pathALL, [img_2d, GT_2d])
            X.append(pathALL)
        bar.update(idx + 1)
    return X
