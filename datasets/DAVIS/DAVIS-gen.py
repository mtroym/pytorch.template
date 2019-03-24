import os
import  numpy as np
import torch
from PIL import Image


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    info = {
        'base_dir': "",
        'train': processing(cacheFilePath, 'train'),
        'val': processing(cacheFilePath, 'val'),
        # 'test': processing(cacheFilePath, 'test')
    }
    print("************* Generating list of data ....**************")
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info


def processing(path, split):
    folders_all = []
    for yr in [2016, 2017]:
        image_set_path = os.path.join(path, 'ImageSet', str(yr), split + '.txt')
        with open(image_set_path, 'r') as f:
            folders = f.read()
            folders = folders.split()
            folders_all += folders
    instance_all = []
    for video in folders_all:
        image_path = os.path.join(path, 'JPEGImages', video)
        gt_path = os.path.join(path, 'Annotations', video)
        ground_truth = np.array(Image.open(gt_path))
        instance_num = np.unique(ground_truth).__len__()
        if np.all(ground_truth == 0):
            continue
        for i in range(instance_num):
            if i == 0:
                continue
            instance_all.append((image_path, gt_path, i))
    return instance_all
