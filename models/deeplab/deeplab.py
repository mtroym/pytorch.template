import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import models.LibTorchLayer as tl
from models.backbones.Xception import Xception
from models.deeplab.ASPP import ASPP

BACKBONE = {'Xception': Xception}


class DeepLab(nn.Module):
    def __init__(self, input_dim, inputsize, backbone, outstride, classes):
        super(DeepLab, self).__init__()
        self.input_dim = input_dim
        if backbone == 'Xception':
            self.backbone = BACKBONE[backbone](input_dim=self.input_dim, outstride=outstride)
        self.x4size = (int(np.ceil(inputsize[0] / 4)), int(np.ceil(inputsize[1] / 4)))
        self.xoutsize = (int(np.ceil(inputsize[0] / outstride)), int(np.ceil(inputsize[1] / outstride)))
        self.ASPP = ASPP(2048, 256, size=self.xoutsize)
        self.inputsize = inputsize
        self.x4conv = tl.SeparableConv2d(128, 128, kernel_size=(1, 1), stride=1, bias=False, bn=False)
        self.decoder_conv = tl.SeparableConv2d(256 + 128, classes, kernel_size=(3, 3), bias=False, bn=False)

    def forward(self, input):
        input, low_level_feat = self.backbone(input)
        input = self.ASPP(input)
        decode_aspp = F.interpolate(input, self.x4size, mode='bilinear', align_corners=True)
        decode_feat = self.x4conv(low_level_feat)
        decode_in = torch.cat((decode_aspp, decode_feat), 1)
        out = self.decoder_conv(decode_in)
        out = F.interpolate(out, self.inputsize, mode='bilinear', align_corners=True)
        return out


class ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=512, depth=256):
        super().__init__()
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
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class Deeplab_v3(nn.Module):
    # in_channel = 3 fine-tune
    def __init__(self, class_number=5, fine_tune=True):
        super().__init__()
        encoder = torchvision.models.resnet50(pretrained=fine_tune)
        self.start = nn.Sequential(encoder.conv1, encoder.bn1,
                                   encoder.relu)

        self.maxpool = encoder.maxpool
        self.low_feature = nn.Sequential(nn.Conv2d(64, 48, 1, 1), nn.ReLU(inplace=True))  # no bn, has bias and relu

        self.layer1 = encoder.layer1  # 256
        self.layer2 = encoder.layer2  # 512
        self.layer3 = encoder.layer3  # 1024
        self.layer4 = encoder.layer4  # 2048

        self.aspp = ASPP(in_channel=2048, depth=256)

        self.conv_cat = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.conv_cat1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.conv_cat2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.score = nn.Conv2d(256, class_number, 1, 1)  # no relu and first conv then upsample, reduce memory

    def forward(self, x):
        size1 = x.shape[2:]  # need upsample input size
        x = self.start(x)
        xm = self.maxpool(x)

        x = self.layer1(xm)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)

        low_feature = self.low_feature(xm)
        size2 = low_feature.shape[2:]
        decoder_feature = F.upsample(x, size=size2, mode='bilinear', align_corners=True)

        conv_cat = self.conv_cat(torch.cat([low_feature, decoder_feature], dim=1))
        conv_cat1 = self.conv_cat1(conv_cat)
        conv_cat2 = self.conv_cat2(conv_cat1)
        score_small = self.score(conv_cat2)
        score = F.upsample(score_small, size=size1, mode='bilinear', align_corners=True)

        return score


def deeplab_v3_50(class_number=5, fine_tune=True):
    model = Deeplab_v3(class_number=class_number, fine_tune=fine_tune)
    return model


def createModel(opt):
    model = Deeplab_v3(class_number=opt.numClasses + 1, fine_tune=False)
    # model = DeepLab(input_dim=opt.input_dim, inputsize=opt.inputSize, backbone='Xception', outstride=8,
    #                 classes=opt.numClasses)
    if opt.GPU:
        model = model.cuda()
    return model

# https://github.com/gengyanlei/deeplab_v3/edit/master/deeplab_v3_50.py
refers_to = "https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py"

if __name__ == '__main__':
    inputsize = (512, 512)
    batch = 1
    C = 1
    net = DeepLab(input_dim=C, inputsize=inputsize, backbone='Xception', outstride=16, classes=5)
    x = torch.randn((batch, C, *inputsize))
    print("input shape", x.shape)
    print("output shape", net(x).shape)
    # print(299.0 / 2 / 2 / 2 / 2)
