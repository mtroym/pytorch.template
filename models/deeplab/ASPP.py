import torch
import torch.nn as nn
import models.torchLayer as tl
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, inplane, outplane, rates=(1, 6, 12, 18), size=None):
        super(ASPP, self).__init__()
        self.at_pool1x1 = tl.SeparableConv2d(inplane, outplane, kernel_size=(1, 1), dilation=rates[0], bn=True)
        self.at_pool3x3_1 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=rates[1], bn=True)
        self.at_pool3x3_2 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=rates[2], bn=True)
        self.at_pool3x3_3 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=rates[3], bn=True)
        self.concate_project_conv = nn.Conv2d(outplane * 5, outplane, kernel_size=(1, 1))
        self.feat_conv = nn.Conv2d(inplane, outplane, kernel_size=(1, 1), bias=False)
        self.concate_project_bn = nn.BatchNorm2d(outplane)
        self.concate_project_drop = nn.Dropout2d(0.1)
        self.size = size
        self.at_poolall = nn.AvgPool2d(kernel_size=self.size)

    def forward(self, x):
        feat = self.at_poolall(x)
        # print(self.size)
        feat = F.interpolate(self.feat_conv(feat), self.size, mode='bilinear', align_corners=True)
        x1conv = self.at_pool1x1(x)
        x3_conv_1 = self.at_pool3x3_1(x)
        x3_conv_2 = self.at_pool3x3_2(x)
        x3_conv_3 = self.at_pool3x3_3(x)
        # print(x1conv.shape, x3_conv_2.shape, feat.shape)
        all = torch.cat((feat, x1conv, x3_conv_1, x3_conv_2, x3_conv_3), 1)
        out = self.concate_project_drop(F.relu(self.concate_project_bn(self.concate_project_conv(all))))
        return out


if __name__ == '__main__':
    net = ASPP(2048, 256, size=(int(299 / 8), int(299 / 8)))
    x = torch.randn((1, 2048, int(299 / 8), int(299 / 8)))
    print(net(x).shape)
