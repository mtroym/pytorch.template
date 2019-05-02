import os

import nibabel as nib
import numpy as np
import torch

from util.progbar import progbar


def exec(opt, cache_file_path):
    data_p = opt.data.replace('3D', '')
    assert os.path.exists(data_p), 'Data directory not found: ' + data_p
    nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限
    training_data_path = os.path.join(data_p, 'train')
    split_ratio = 0.8
    patients = [path for path in os.listdir(training_data_path) if path.startswith('Patient')]
    print('!> basedir = %s' % training_data_path)
    train_pa = patients[:int(len(patients) * split_ratio)]
    val_pa = patients[int(len(patients) * split_ratio):]
    print("************* Generating list of data ....**************")
    info = {
        'basedir': training_data_path,
        'val': load_paths(opt, training_data_path, val_pa, 'val'),
        'train': load_paths(opt, training_data_path, train_pa, 'train')
    }
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cache_file_path))
    torch.save(info, cache_file_path)
    return info


def load_paths(opt, base, pa, split):
    # ignoring the file with no label.
    print('\t-> processing {} data'.format(split))
    X = []
    Y = []
    Z = []
    bar = progbar(len(pa), width=opt.barwidth)
    for idx, patients in enumerate(pa):
        # file contains GT.nii // patient_xx.nii.gz
        gt_path = os.path.join(base, patients, 'GT.nii.gz')
        img_path = os.path.join(base, patients, patients + '.nii.gz')
        gt = nib.load(gt_path).get_data()
        for i in range(gt.shape[2]):
            Z.append(((patients, i), gt_path, img_path))
        for i in range(gt.shape[0]):
            if i <  85 or i  > 401:
                continue
            X.append(((patients, i), gt_path, img_path))
        for i in range(gt.shape[1]):
            if i < 162 or i > 413:
                continue
            Y.append(((patients, i), gt_path, img_path))
        bar.update(idx + 1)
    out = dict()
    out['X'] = X
    out['Y'] = Y
    out['Z'] = Z
    return X, Y, Z
