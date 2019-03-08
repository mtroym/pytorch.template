import os

import nibabel as nib
import numpy as np
import torch

from util.progbar import progbar

def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限
    training_data_path = os.path.join(opt.data, 'train')
    split_ratio = 0.8
    patients = [path for path in os.listdir(training_data_path) if path.startswith('Patient')]
    print('!> basedir = %s' % training_data_path)
    train_pa = patients[:int(len(patients) * split_ratio)]
    val_pa = patients[int(len(patients) * split_ratio):]
    print("************* Generating list of data ....**************")
    info = {
        'basedir': training_data_path,
        'val': loadPaths(opt, training_data_path, val_pa, 'val'),
        'train': loadPaths(opt, training_data_path, train_pa, 'train')
    }
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info


def loadPaths(opt, base, pa, split):
    print('\t-> processing {} data'.format(split))
    X = []
    bar = progbar(len(pa), width=opt.barwidth)
    for idx, patients in enumerate(pa):
        # file contains GT.nii // patient_xx.nii.gz
        GT_path = os.path.join(base, patients, 'GT.nii.gz')
        img_path = os.path.join(base, patients, patients + '.nii.gz')
        img = nib.load(img_path).get_data()
        for i in range(img.shape[2]):
            if np.sum(img[:,:,i]) == 0:
                continue
            X.append(((patients, i), GT_path, img_path))
        bar.update(idx + 1)
    return X
