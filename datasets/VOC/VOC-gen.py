import os

import torch
from tqdm import tqdm


def loadPaths(opt, path, split):
    txt_name = split + '.txt'
    split_path = os.path.join(path, 'ImageSets', 'Segmentation', txt_name)
    with open(split_path, 'r') as f:
        split_list = f.readlines()

    base_dir_path = os.path.join(path, 'JPEGImages')
    gt_dir_path = os.path.join(path, 'SegmentationClass')

    Paths = []
    for split_slice in tqdm(split_list):
        img_path = os.path.join(base_dir_path, split_slice.replace('\n', '')+ '.jpg')
        gt_path = os.path.join(gt_dir_path, split_slice.replace('\n', '') + '.png')
        Paths.append((img_path, gt_path))
    return Paths


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    baseDir = os.path.join(opt.data, 'VOCdevkit', 'VOC2012')
    print('!> basedir = %s' % baseDir)
    print("************* Generating list of data ....**************")
    info = {
        'basedir': baseDir,
        'val': loadPaths(opt, baseDir, 'val'),
        'train': loadPaths(opt, baseDir, 'train'),
        'trainval': loadPaths(opt, baseDir, 'trainval'),
    }
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info