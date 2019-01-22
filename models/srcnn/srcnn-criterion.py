import torch.nn as nn
from skimage.measure import compare_psnr, compare_ssim
from collections import defaultdict
def initCriterion(criterion, model):
    pass


def createCriterion(opt, model):
    criterion = nn.MSELoss()
    return criterion


def accuracy(outputs, labels):
    N, _, _, _ = outputs.shape
    psnr = 0
    for i in range(N):
        psnr += compare_psnr(labels[i], outputs[i])
    return psnr / N


def ssim(outputs, labels):
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        ssim += compare_ssim(labels[i], outputs[i], win_size=3, multichannel=True)
    return ssim / N

METRICS = {
    'PSNR': accuracy,
    'SSIM': ssim,
}

def createMetrics(opt, model):
    print("=> create metrics: ", end="")
    metrics = {}
    for metric in opt.metrics:
        print(metric, end=" ,")
        metrics[metric] = METRICS[metric]
    print()
    return metrics