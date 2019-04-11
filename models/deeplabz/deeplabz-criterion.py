import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn

from criterions.lovasz_loss import LovaszSoftmax


# LOSS : return total_loss(float), loss_dict(dict()) -> {'total': loss, 'singleLoss1': loss}


def initCriterion(criterion, model):
    pass


def createCriterion(opt, model):
    criterion = Custom_Criterion(ignore=-100)
    return criterion


class Custom_Criterion(nn.Module):
    def __init__(self, ignore=-100):
        super(Custom_Criterion, self).__init__()
        self.ignore = ignore
        self.Lovasz = LovaszSoftmax(only_present=True, ignore_index=self.ignore)
        self.CELoss = nn.CrossEntropyLoss(ignore_index=self.ignore)

    def forward(self, x, y):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          only_present: average only on classes present in ground truth
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        loss_lovasz = self.Lovasz(x, y)
        loss_crossentropy = self.CELoss(x, y)
        total_loss = loss_lovasz + loss_crossentropy
        loss = {'loss': float(total_loss.detach().cpu().numpy()),
                'lovasz': float(loss_lovasz.detach().cpu().numpy()),
                'crossetropy': float(loss_crossentropy.detach().cpu().numpy())}
        return total_loss, loss


class mIoU(nn.Module):
    def __init__(self, C, ignore=-100, per_image=False):
        super(mIoU, self).__init__()
        self.C = C
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, x, y):
        ious = compute_ious(x, y, classes=self.C, ignore_index=self.ignore, only_present=True)
        if np.all(np.isnan(np.array(list(ious.values())))):
            return 0.0
        return {'mIoU': np.nanmean(np.array(list(ious.values())))}


class IoU(nn.Module):
    def __init__(self, C, ignore=-100, per_image=False):
        super(IoU, self).__init__()
        self.C = C
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, x, y):
        dict_iou = compute_ious(x, y, classes=self.C, ignore_index=self.ignore, only_present=True)
        return dict_iou


class Hausdorff(nn.Module):
    def __init__(self, C, ignore=-100):
        super(Hausdorff, self).__init__()
        self.C = C
        self.ignore = ignore
        self.hausdorffcomputer = [sitk.HausdorffDistanceImageFilter() for _ in range(C)]

    def forward(self, x, y):
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
            self.hausdorffcomputer[i].Execute(P == i, GT == i)
            quality["avg_Hausdorff_class" + str(i)] = self.hausdorffcomputer[i].GetAverageHausdorffDistance()
            total_val += quality["avg_Hausdorff_class" + str(i)]
            count += 1
        quality["AVG_Hausdorff"] = total_val / (count + 1e-10)
        return quality


class DiceCoeff(nn.Module):
    def __init__(self, C, ignore=-100, only_present=True):
        super(DiceCoeff, self).__init__()
        self.C = C
        self.ignore = ignore
        self.only_present = only_present

    def forward(self, pred, label):
        pred[label == self.ignore] = 0
        quality = {}
        count = 0.0
        for i in range(self.C):
            if i == self.ignore:
                continue
            label_c = label == i
            pred_c = pred == i
            if self.only_present and torch.sum(label_c) == 0:
                quality['Dice#' + str(i)] = np.nan
                continue
            overlap = (pred_c & label_c).sum()
            unions = pred_c.sum() + label_c.sum()
            if unions != 0:
                quality["Dice#" + str(i)] = 2 * float(overlap) / float(unions + 1e-10)
                count += 1
            else:
                quality["Dice#" + str(i)] = np.nan
        return quality


numClasses = 21

METRICS = {
    'mIoU': mIoU,
    'Dice': DiceCoeff,
    'Hausdorff': Hausdorff,
    'IoU': IoU
}


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    pred[label == ignore_index] = 0
    ious = {}
    for c in range(classes):
        if c == ignore_index:
            # ious['IoU#' + str(c)] = np.nan
            continue
        label_c = label == c
        pred_c = pred == c
        if only_present and torch.sum(label_c) == 0:
            ious['IoU#' + str(c)] = np.nan
            continue
        intersection = (pred_c & label_c).sum()
        union = (pred_c | label_c).sum()
        if union != 0:
            ious['IoU#' + str(c)] = float(intersection) / float(union)
        else:
            ious['IoU#' + str(c)] = np.nan
    return ious


class Metrics(nn.Module):
    def __init__(self, opt, l, ignore_index=-100):
        super(Metrics, self).__init__()
        self.metrics = {}
        self.name = []
        for m in l:
            self.name.append(m)
            self.metrics[m] = METRICS[m](opt.numClasses, ignore=ignore_index)

    def forward(self, x, y):
        out = dict()
        for m in self.metrics:
            metrics_value = self.metrics[m](x, y)
            if isinstance(metrics_value, dict):
                for key in metrics_value:
                    out[key] = metrics_value[key]
            else:
                assert isinstance(metrics_value, float), 'Error of the metric value type {}'.format(type(metrics_value))
                out[m] = metrics_value
        return out


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

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
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def createMetrics(opt, model):
    metrics = Metrics(opt, opt.metrics, 0)
    mstr = ('[' + ', '.join(opt.metrics) + ']')
    print("=> create metrics: " + mstr)
    return metrics


if __name__ == '__main__':
    # import opts
    #
    # opt = opts.parse()
    # M = Metrics(opt, ['mIoU'], ignore_index=0)
    # x = np.ones((10, 30, 30))
    # y = np.ones((10, 30, 30))
    # for m in M.name:
    #     print(M[m](x, y))
    a = dict()
    a['x'] = 1
    a['y'] = 2
    a['z'] = np.nan
    for k in a:
        print(k)
    a.update({'m': np.nanmean(np.array(list(a.values())))})
    print(a)
