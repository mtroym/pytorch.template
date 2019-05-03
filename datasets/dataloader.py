import sys

sys.path.append("..")
import datasets.init as datasets
from torch.utils.data.dataloader import *


class myDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False):
        DataLoader.__init__(self, dataset, batch_size, shuffle, sampler, batch_sampler,
                            num_workers, collate_fn, pin_memory, drop_last)


def create(opt):
    loaders = []
    for split in ['train', 'val']:
        dataset = datasets.create(opt, split)
        if split == 'train':
            if opt.dataset == 'segTHOR3D':
                loaders.append(myDataLoader(dataset=dataset[0], batch_size=opt.batchSize, shuffle=True,
                                            num_workers=opt.nThreads, pin_memory=opt.GPU))
                loaders.append(myDataLoader(dataset=dataset[1], batch_size=opt.batchSize, shuffle=True,
                                        num_workers=opt.nThreads, pin_memory=opt.GPU))
                loaders.append(myDataLoader(dataset=dataset[2], batch_size=opt.batchSize, shuffle=True,
                                            num_workers=opt.nThreads, pin_memory=opt.GPU))
            else:
                loaders.append(myDataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=True,
                                            num_workers=opt.nThreads, pin_memory=opt.GPU))
        elif split == 'val':
            if opt.dataset == 'segTHOR3D':
                loaders.append(myDataLoader(dataset=dataset[0], batch_size=opt.batchSize, shuffle=False,
                                            num_workers=opt.nThreads, pin_memory=opt.GPU))
                loaders.append(myDataLoader(dataset=dataset[1], batch_size=opt.batchSize, shuffle=False,
                                        num_workers=opt.nThreads, pin_memory=opt.GPU))
                loaders.append(myDataLoader(dataset=dataset[2], batch_size=opt.batchSize, shuffle=False,
                                            num_workers=opt.nThreads, pin_memory=opt.GPU))
            else:
                loaders.append(myDataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=False,
                                        num_workers=opt.nThreads, pin_memory=opt.GPU))
    if opt.dataset== 'segTHOR3D':
        return loaders[0:3], loaders[3:]
    else:
        return loaders[0], loaders[1]
