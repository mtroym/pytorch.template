import os
import numpy as np
import torch
from PIL import Image
from util.progbar import progbar


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    info = {
        'base_dir': "",
        'train': processing(opt.data, 'train'),
        'val': processing(opt.data, 'val'),
        # 'test': processing(cacheFilePath, 'test')
    }
    print("************* Generating list of data ....**************")
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info


def processing(path, split):
    folders_all = []
    print("=> start collecting {} data....".format(split))
    for yr in [2016, 2017]:
        image_set_path = os.path.join(path, 'ImageSets', str(yr), split + '.txt')
        with open(image_set_path, 'r') as f:
            folders = f.read()
            folders = folders.split()
            folders_all += folders
    instance_all = []
    bar = progbar(len(folders_all), width=40)
    for idx, video in enumerate(folders_all):
        bar.update(idx + 1)
        image_path = os.path.join(path, 'JPEGImages', '480p', video)
        gt_path = os.path.join(path, 'Annotations', '480p', video)
        for frame in os.listdir(gt_path):
            if frame.endswith('.png'):
                ground_truth = np.array(Image.open(os.path.join(gt_path, frame)))
                instance_num = np.unique(ground_truth).__len__()
            if np.all(ground_truth == 0):
                continue
            for i in range(instance_num):
                if i == 0:
                    continue
                gt_path_ = os.path.join(gt_path, frame)
                image_path_ = os.path.join(image_path, frame.replace('png', 'jpg'))
                instance_all.append((image_path_, gt_path_, video, frame, i))
    return instance_all
