import torch
import torch.nn.functional as f
from torch import nn
from models.LibTorchLayer.ConvLSTM2d import *


all = ["ConvBlock", "SeparableConv2d", "ConvLSTM2d", "BiConvLSTM2d"]
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu',
                 norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm != "None":
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != "None":
            return self.act(out)
        else:
            return out





class BilinearUpsampler(nn.Module):
    def __init__(self, size):
        super(BilinearUpsampler, self).__init__()
        self.size = size

    def forward(self, input):
        return f.interpolate(input, self.size, mode='bilinear', align_corners=True)


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

