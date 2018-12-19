import torch
import os
import math
import numpy as np
import subprocess
import torchvision

def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data

    print("=> Generating list of data")

    info = {'basedir' : opt.data}

    torch.save(info, cacheFile)
    return info
