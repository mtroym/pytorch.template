import numpy as np
import torch.nn as nn

import criterions.LovaszSoftmax as L
import SimpleITK as sitk


def initCriterion(criterion, model):
    pass


def createCriterion(opt, model):
    criterion = Criterion(ignore=0)
    return criterion



class Criterion(nn.Module):
    def __init__(self, ignore=-100):
        super(Criterion, self).__init__()
        self.ignore = ignore

    def forward(self, x, y):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          only_present: average only on classes present in ground truth
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        return L.lovasz_softmax(x, y, only_present=False, per_image=False, ignore=self.ignore)


# CELOSS.
class _Criterion(nn.Module):
    def __init__(self, ignore=-100):
        super(_Criterion, self).__init__()
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


class Hausdorff(nn.Module):
    def __init__(self, C, ignore=-100):
        super(Hausdorff, self).__init__()
        self.C = C
        self.ignore = ignore
        self.hausdorffcomputer = [sitk.HausdorffDistanceImageFilter() for _ in range(C)]

    def forward(self, x, y):
        print(x, y)
        quality = {}
        total_val = 0.0
        count = 0.0
        x = x * 0
        P = sitk.GetImageFromArray(x.astype(np.int32), isVector=False)
        GT = sitk.GetImageFromArray(y.astype(np.int32), isVector=False)
        for i in range(self.C):
            if i == self.ignore:
                continue
            # P = sitk.GetImageFromArray((x == i).astype(np.int))
            # GT = sitk.GetImageFromArray((y == i).astype(np.int))
            self.hausdorffcomputer[i].Execute(P==i, GT==i)
            quality["avg_Hausdorff_class" + str(i)] = self.hausdorffcomputer[i].GetAverageHausdorffDistance()
            total_val += quality["avg_Hausdorff_class" + str(i)]
            count += 1
        quality["AVG_Hausdorff"] = total_val / (count + 1e-10)
        return quality

class DiceCoeff(nn.Module):
    def __init__(self, C, ignore=-100, per_image=False):
        super(DiceCoeff, self).__init__()
        self.C = C
        self.ignore = ignore
        self.per_image = per_image

        self.dicecomputer = [sitk.LabelOverlapMeasuresImageFilter() for _ in range(C)]

    def forward(self, x, y):
        quality = {}
        total_val = 0.0
        count = 0.0
        x = x * 0
        for i in range(self.C):
            if i == self.ignore:
                continue
            P = sitk.GetImageFromArray((x.astype(np.int) == i).astype(np.int))
            GT = sitk.GetImageFromArray((y.astype(np.int) == i).astype(np.int))
            self.dicecomputer[i].Execute(P, GT)
            quality["Dice_" + str(i)] = self.dicecomputer[i].GetDiceCoefficient()
            total_val += quality["Dice_" + str(i)]
            count += 1
        quality["AVG_Dice"] = total_val / (count + 1e-10)
        return quality

METRICS = {
    'mIoU': mIoU(5, ignore=0),
    'Dice': DiceCoeff(5, ignore=0),
    'Hausdorff': Hausdorff(5, ignore=0)
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

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self, ignore_index=None):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if ignore_index is not None:
            MIoU = np.delete(MIoU, ignore_index, 0)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return IoU


    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)





def createMetrics(opt, model):
    metrics = Metrics(opt.metrics)
    mstr = ('[' + ', '.join(opt.metrics) + ']')
    print("=> create metrics: " + mstr)
    return metrics


if __name__ == '__main__':
    M = Metrics(['mIoU'])
    for m in M.name:
        M[m](0, 1)
