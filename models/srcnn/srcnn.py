import os
import torch
import torch.nn as nn
import models.torchLayer as tl


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.num_channels = opt.numChannels
        self.dropout_rate = opt.dropoutRate

        self.layers = torch.nn.Sequential(
            tl.ConvBlock(3, self.num_channels, 9, 1, 4, norm="None"),  # 144*144*64 # conv->batchnorm->activation
            # Kernel = 9, stride = 1,  padding = 4 for the same size output.
            tl.ConvBlock(self.num_channels, self.num_channels // 2, 1, 1, 0, norm="None"),  # 144*144*32
            # Kernel = 1, stride = 1,  padding = 0 mapping.
            tl.ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation="None", norm="None")  # 144*144*3
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
