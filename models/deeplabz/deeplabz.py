import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.LibTorchLayer as tl
from models.backbones import build_backbone
from models.backbones import Xception
from models.deeplab.ASPP import ASPP
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplab.ASPP import build_aspp
from models.deeplab.Decoder import build_decoder

# import torch
# import torch.nn.functional as F
# import torchvision
# from torch import nn

SynchronizedBatchNorm2d = nn.BatchNorm2d

class _ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=512, depth=256):
        super(_ASPP).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1), nn.ReLU(inplace=True))

        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
                                           nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
                                            nn.ReLU(inplace=True))

        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net




class DeepLab(nn.Module):
    def __init__(self, input_dim=3, backbone='Resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(input_dim, backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, with_height=True)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, height=None):
        """
        :param input: torch tensor
        :param height: double tensor in [-0.5, 0.5]
        :return:
        """
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat, height)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


# https://github.com/gengyanlei/deeplab_v3/edit/master/deeplab_v3_50.py
refers_to = "https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py"


def createModel(opt):
    model = DeepLab(input_dim=opt.input_dim, backbone=opt.backbone, output_stride=opt.outStride, num_classes=opt.numClasses, sync_bn=False,
                    freeze_bn=False)
    if opt.GPU:
        model = model.cuda()
    return model
