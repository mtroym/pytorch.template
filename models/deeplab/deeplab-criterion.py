import numpy as np
import torch.nn as nn

import criterions.LovaszSoftmax as L


def initCriterion(criterion, model):
    pass


def createCriterion(opt, model):
    criterion = Criterion(ignore=0)
    return criterion


class Criterion(nn.Module):
    def __init__(self, ignore=None):
        super(Criterion, self).__init__()
        ignore = -100 if ignore is None else ignore
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore)

    def forward(self, x, y):
        return self.criterion(x, y)


class mIoU(nn.Module):
    def __init__(self, C, ignore=-100, per_image=False):
        super(mIoU, self).__init__()
        self.C = C
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, *input):
        return np.mean(L.iou(preds=input[0], labels=input[1], C=self.C,
                             EMPTY=0.0, ignore=self.ignore,
                             per_image=self.per_image))


# def mIoU(preds, labels, ignore=0):
#     ious = L.iou(preds, labels, 5, EMPTY=0.0, ignore=ignore, per_image=False)
#     return np.mean(ious)


METRICS = {
    'mIoU': mIoU(5, 0),
    'IoUs': lambda preds, labels: np.array(L.iou(preds, labels, 5, EMPTY=0.0, ignore=0, per_image=False))
}


class Metrics(nn.Module):
    def __init__(self, l):
        super(Metrics, self).__init__()
        self.metrics = {}
        self.name = []
        for m in l:
            self.name.append(m)
            self.metrics[m] = METRICS[m]

    def __getitem__(self, item):
        return self.metrics[item]

    def forward(self, *input):
        out = []
        for m in self.metrics:
            evVal = self.metrics[m](input)
            out.append((m, evVal))
        return out


def createMetrics(opt, model):
    metrics = Metrics(opt.metrics)
    mstr = ('[' + ', '.join(opt.metrics) + ']')
    print("=> create metrics: " + mstr)
    return metrics


if __name__ == '__main__':
    M = Metrics(['mIoU'])
    for m in M.name:
        M[m](0, 1)
