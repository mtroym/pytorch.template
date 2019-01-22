import os
import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm != "None":
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != "None":
            return self.act(out)
        else:
            return out


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.num_channels = opt.numChannels
        self.dropout_rate = opt.dropoutRate

        self.layers = torch.nn.Sequential(
            ConvBlock(3, self.num_channels, 9, 1, 4, norm="None"),  # 144*144*64 # conv->batchnorm->activation
            # Kernel = 9, stride = 1,  padding = 4 for the same size output.
            ConvBlock(self.num_channels, self.num_channels // 2, 1, 1, 0, norm="None"),  # 144*144*32
            # Kernel = 1, stride = 1,  padding = 0 mapping.
            ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation="None", norm="None")  # 144*144*3
            # Kernel = 5, stride = 1,  padding = 2 for the same size output.
        )

    def forward(self, s):
        out = self.layers(s)
        return out


def createModel(opt):
    rootPath = (os.path.dirname(os.path.abspath('.')))
    model = Net(opt)
    if opt.GPU:
        model = model.cuda()
    return model
