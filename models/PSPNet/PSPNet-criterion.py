import torch.nn as nn
import torch
import numpy as np
import criterions.lovasz_loss


def createCriterion(opt, model):
    return Custom_Criterion(ignore=-100)


class Custom_Criterion(nn.Module):
    def __init__(self, ignore=-100):
        super(Custom_Criterion, self).__init__()
        self.ignore = ignore
        self.Lovasz = criterions.lovasz_loss.LovaszSoftmax(only_present=True, ignore_index=self.ignore)
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


def createMetrics(opt, model):
    return IoU


class IoU(nn.Module):
    def __init__(self, C, ignore=-100, per_image=False):
        super(IoU, self).__init__()
        self.C = C
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, x, y):
        dict_iou = compute_ious(x, y, classes=self.C, ignore_index=self.ignore, only_present=True)
        return dict_iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    pred[label == ignore_index] = 0
    ious = {}
    for c in range(classes):
        if c == ignore_index:
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
