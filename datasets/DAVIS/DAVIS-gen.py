import os

import torch


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    info = {
        'base_dir': "",
        'train': "",
        'val': ""
    }
    print("************* Generating list of data ....**************")
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info
