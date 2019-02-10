from numpy import log
import torch.nn as nn

import models.torchLayer as tl

__all__ = ['Xception']


class Xception_Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, dilation=1,
                 start_with_relu=True, grow_first=True):
        super(Xception_Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(tl.SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1,
                                          dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(tl.SeparableConv2d(filters, filters, kernel_size=3, stride=1,
                                          dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(tl.SeparableConv2d(filters, out_filters, kernel_size=3, stride=1,
                                          dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=strides, padding=3 // 2))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception_entry_flow(nn.Module):
    def __init__(self, input_channel, out_channel, last_stride=2):
        super(Xception_entry_flow, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32,
                               kernel_size=(3, 3), stride=2, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        # ------------
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        # ------------
        self.layer3 = Xception_Block(in_filters=64, out_filters=128, strides=2,
                                     reps=2, start_with_relu=False, grow_first=True)
        self.layer4 = Xception_Block(in_filters=128, out_filters=256, strides=2,
                                     reps=2, start_with_relu=False, grow_first=True)
        self.layer5 = Xception_Block(in_filters=256, out_filters=out_channel, strides=last_stride,
                                     reps=2, start_with_relu=False, grow_first=True)

    def forward(self, inputs):
        inputs = self.relu1(self.bn1(self.conv1(inputs)))
        inputs = self.relu2(self.bn2(self.conv2(inputs)))
        # print(inputs.shape)
        inputs = self.layer3(inputs)
        mid = inputs
        inputs = self.layer4(inputs)
        inputs = self.layer5(inputs)
        return inputs, mid


class Xception_middle_flow(nn.Module):
    def __init__(self, filters, rep, dilation):
        super(Xception_middle_flow, self).__init__()
        middle_path = []
        for i in range(rep):
            middle_path.append(Xception_Block(in_filters=filters, out_filters=filters,
                                              reps=3, dilation=dilation,
                                              start_with_relu=True, grow_first=False))
        self.net = nn.Sequential(*middle_path)

    def forward(self, x):
        return self.net(x)


class Xception_exit_flow(nn.Module):
    def __init__(self, in_filters, mid, out_filters, dilations=(1, 1, 1, 1)):
        super(Xception_exit_flow, self).__init__()

        self.block12 = Xception_Block(in_filters=in_filters, out_filters=mid, reps=2,
                                      dilation=dilations[0], start_with_relu=True, grow_first=False)
        mid2 = (in_filters + out_filters) // 2
        self.conv3 = tl.SeparableConv2d(inplanes=mid, planes=mid2, kernel_size=3, stride=1, dilation=dilations[1])
        self.bn3 = nn.BatchNorm2d(mid2)
        self.conv4 = tl.SeparableConv2d(inplanes=mid2, planes=mid2, kernel_size=3, stride=1, dilation=dilations[2])
        self.bn4 = nn.BatchNorm2d(mid2)
        self.conv5 = tl.SeparableConv2d(inplanes=mid2, planes=out_filters,
                                        kernel_size=3, stride=1, dilation=dilations[3])
        self.bn5 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.block12(x)
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = self.bn5(self.conv5(x))
        return x


class Xception(nn.Module):
    def __init__(self, outstride=None):
        super(Xception, self).__init__()
        # entry_flow_conv
        config_16 = {
            'entry_last_stride': 2,
            'middle_dilation': 1,
            'exit_dilation': (1, 2, 2, 2)
        }
        config_8 = {
            'entry_last_stride': 1,
            'middle_dilation': 2,
            'exit_dilation': (2, 4, 4, 4)
        }
        config_default = {
            'entry_last_stride': 2,
            'middle_dilation': 1,
            'exit_dilation': (1, 1, 1, 1)
        }
        configs = [config_8, config_16]
        if outstride is None:
            config = config_default
        else:
            config = configs[int(log(outstride) / log(2)) - 3]

        self.entry = Xception_entry_flow(1, 728, last_stride=config['entry_last_stride'])
        self.middle = Xception_middle_flow(728, 8, dilation=config['middle_dilation'])
        self.exit = Xception_exit_flow(728, 1024, 2048, dilations=config['exit_dilation'])
        # in

    def forward(self, x):
        x, low_level_feat = self.entry(x)
        x = self.middle(x)
        out = self.exit(x)
        return out, low_level_feat
