import os
import torch
import importlib
import subprocess

def isvalid(opt, cachePath):
    info = torch.load(cachePath)
    if info['basedir'] != opt.data:
        return False
    return True

def create(opt, split):
    cachePath = os.path.join(opt.gen, opt.dataset + '.pth.tar')

    if not os.path.exists(cachePath) or not isvalid(opt, cachePath):
        script = opt.dataset + '-gen'
        gen = importlib.import_module('datasets.' + script)
        gen.exec(opt, cachePath)

    info = torch.load(cachePath)
    dataset = importlib.import_module('datasets.' + opt.dataset)
    return dataset.getInstance(info, opt, split)
