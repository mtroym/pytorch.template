# Pyramid Dilated Convolution (PDC) module

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.LibTorchLayer as tl


class PDC(nn.Module):
    def __init__(self, inplane, outplane, size=None):
        super(PDC, self).__init__()
        self.size = size
        self.at_pool1 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=2, bn=True)
        self.at_pool2 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=4, bn=True)
        self.at_pool3 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=8, bn=True)
        self.at_pool4 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=16, bn=True)
        # TODO: cat.

    def forward(self, x):
        """
        :param x: (B, T, C, H, W)
        :return: (B, T, CC, H, W)
        """
        seq_Len = x.size(1)
        out_temp = []
        for t in range(seq_Len):
            out_temp.append(self._forward(x[:,t,:,:,:]))
        out = torch.stack(out_temp, dim=1)
        return out

    def _forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: (B, CC, H, W)
        """
        x3_conv_0 = self.at_pool1(x)
        x3_conv_1 = self.at_pool2(x)
        x3_conv_2 = self.at_pool3(x)
        x3_conv_3 = self.at_pool4(x)
        at_all = torch.cat((x, x3_conv_0, x3_conv_1, x3_conv_2, x3_conv_3), 1)
        out = F.interpolate(at_all, self.size, mode='bilinear', align_corners=True)
        return out

if __name__ == '__main__':
    T = 5
    B = 3
    C = 512
    H = 20
    W = 20
    inputsize = (H, W)
    batch = 1
    net = PDC(inplane=C, outplane=128, size=inputsize)
    x = torch.randn((B, T, C, H, W))
    print(net(x).shape)
    # print(299.0 / 2 / 2 / 2 / 2)
