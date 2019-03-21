import os

import nibabel as nib
import numpy as np
import torch

from util.progbar import progbar


def exec(opt, cache_file_path):
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
    bar = progbar(len(pa), width=opt.barwidth)
    for idx, patients in enumerate(pa):
        # file contains GT.nii // patient_xx.nii.gz
        gt_path = os.path.join(base, patients, 'GT.nii.gz')
        img_path = os.path.join(base, patients, patients + '.nii.gz')
        gt = nib.load(gt_path).get_data()
        for i in range(gt.shape[2]):
            if np.sum(gt[:, :, i]) == 0 and split == 'train':
                continue
            X.append(((patients, i), gt_path, img_path))
            if i == gt.shape[2] - 1 and (split == 'val' or split == 'test'):  # last slice
                X.append(((patients, -1), None, None))
        bar.update(idx + 1)
    return X
