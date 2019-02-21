import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import criterions.LovaszSoftmax as L


def initCriterion(criterion, model):
    pass


def createCriterion(opt, model):
    criterion = Criterion(ignore=None)
    return criterion


class Criterion(nn.Module):
    def __init__(self, ignore=None):
        super(Criterion, self).__init__()
        # self.criterion = lambda x, y: L.lovasz_softmax(x, y, ignore=ignore)
        ignore = -100 if ignore is None else ignore
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore)

    def forward(self, x, y):
        return self.criterion(x, y)


def mIoU(preds, labels, ignore=0):
    ious = L.iou(preds, labels, 5, EMPTY=0.0, ignore=ignore, per_image=False)
    return np.mean(ious)


METRICS = {
    'mIoU': lambda preds, labels: mIoU(preds, labels),
    'IoUs': lambda preds, labels: np.array(L.iou(preds, labels, 5, EMPTY=0.0, ignore=0, per_image=False))
}


def createMetrics(opt, model):
    print("=> create metrics: ", end="")
    metrics = {}
    for metric in opt.metrics:
        print(metric, end=", ")
        metrics[metric] = METRICS[metric]
    print()
    return metrics


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction=reduction)

    def forward(self, logits, targets):
        probs = F.softmax(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(-logits.abs()))
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / (
                (w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class DiceScore(nn.Module):
    def __init__(self, threshold=0.5):
        super(DiceScore, self).__init__()
        self.threshold = threshold

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        predicts = (probs.view(num, -1) > self.threshold).float()
        labels = labels.view(num, -1)
        intersection = (predicts * labels)
        score = 2. * (intersection.sum(1)) / (predicts.sum(1) + labels.sum(1))
        return score.mean()


def dice_score_np(predicted_masks, labels):
    assert len(predicted_masks.shape) >= 3
    num = predicted_masks.shape[0]

    predicted_masks = predicted_masks.reshape(num, -1).astype(float)
    assert predicted_masks.min() == 0 and predicted_masks.max() == 1.0
    labels = labels.reshape(num, -1)
    assert labels.min() == 0 and labels.max() == 1.0
    intersection = (predicted_masks * labels)
    assert not np.any((predicted_masks.sum(1) + labels.sum(1)) == 0)
    score = 2. * (intersection.sum(1)) / (predicted_masks.sum(1) + labels.sum(1))
    assert len(score.shape) == 1
    return score.mean()


class CombinedLoss(nn.Module):
    def __init__(self, is_weight=True, is_log_dice=False):
        super(CombinedLoss, self).__init__()
        self.is_weight = is_weight
        self.is_log_dice = is_log_dice
        if self.is_weight:
            self.weighted_bce = WeightedBCELoss2d()
            self.soft_weighted_dice = WeightedSoftDiceLoss()
        else:
            self.bce = BCELoss2d()
            self.soft_dice = SoftDiceLoss()

    def forward(self, logits, labels):
        size = logits.size()
        assert size[1] == 1, size
        logits = logits.view(size[0], size[2], size[3])
        labels = labels.view(size[0], size[2], size[3])
        if self.is_weight:
            batch_size, H, W = labels.size()
            if H == 128:
                kernel_size = 11
            elif H == 256:
                kernel_size = 21
            elif H == 512:
                kernel_size = 21
            elif H == 1024:
                kernel_size = 41
            elif H == 1280:
                kernel_size = 51
            else:
                raise ValueError('Unknown height')

            a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2,
                             stride=1)
            ind = a.ge(0.01) * a.le(0.99)
            ind = ind.float()
            weights = Variable(torch.tensor.torch.ones(a.size())).cuda()

            w0 = weights.sum()
            weights += ind * 2
            w1 = weights.sum()
            weights = weights / w1 * w0

            bce_loss = self.weighted_bce(logits, labels, weights)
            dice_loss = self.soft_weighted_dice(logits, labels, weights)
        else:
            bce_loss = self.bce(logits, labels)
            dice_loss = self.soft_dice(logits, labels)

        if self.is_log_dice:
            l = bce_loss - (1 - dice_loss).log()
        else:
            l = bce_loss + dice_loss
        return l, bce_loss, dice_loss


def combined_loss(logits, labels, is_weight=True, is_log_dice=False):
    size = logits.size()
    assert size[1] == 1, size
    logits = logits.view(size[0], size[2], size[3])
    labels = labels.view(size[0], size[2], size[3])
    if is_weight:
        batch_size, H, W = labels.size()
        if H == 128:
            kernel_size = 11
        elif H == 256:
            kernel_size = 21
        elif H == 512:
            kernel_size = 21
        elif H == 1024:
            kernel_size = 41
        elif H == 1280:
            kernel_size = 51
        else:
            raise ValueError('Unknown height')

        a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = Variable(torch.Tensor.ones(a.size())).cuda()

        w0 = weights.sum()
        weights += ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        bce_loss = WeightedBCELoss2d().cuda()(logits, labels, weights)
        dice_loss = WeightedSoftDiceLoss().cuda()(logits, labels, weights)
    else:
        bce_loss = BCELoss2d().cuda()(logits, labels)
        dice_loss = SoftDiceLoss().cuda()(logits, labels)

    if is_log_dice:
        l = bce_loss - (1 - dice_loss).log()
    else:
        l = bce_loss + dice_loss
    return l, bce_loss, dice_loss
