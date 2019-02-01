import os

import torch


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    print(">>>=====> Generating list of dat ....")
    info = {
        'basedir': opt.data,
        'test': loadPaths(opt.data, 'test'),
        'val': loadPaths(opt.data, 'val'),
        'train': loadPaths(opt.data, 'train')
    }
    torch.save(info, cacheFilePath)
    return info


def loadPaths(base, split):
    path_input = os.path.join(base, split + "_blur")
    path_target = os.path.join(base, split + "_clear")
    list_input, list_target = listdir_(path_input), listdir_(path_target)
    assert list_input[10].split("/")[-1] == list_target[10].split("/")[-1], \
        "WARNING, sanity check in data loader failed."
    return list_input, list_target


def listdir_(path):
    filenames = os.listdir(path)  # label
    filenames = [os.path.join(path, f) for f in filenames if f.endswith('.jpg')]  # label
    return filenames
