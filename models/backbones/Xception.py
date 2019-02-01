import torch.nn as nn

import models.torchLayer as tl

__all__ = ['Xception']


class Xception_Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Xception_Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(tl.SeparableConv2d(in_filters, out_filters, 3, stride=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(tl.SeparableConv2d(filters, filters, 3, stride=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(tl.SeparableConv2d(in_filters, out_filters, 3, stride=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
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
    def __init__(self, step1, step2, step3, step4, step5):
        super(Xception_entry_flow, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=step1,
                               kernel_size=(3, 3), stride=2, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        # ------------
        self.conv2 = nn.Conv2d(in_channels=step1, out_channels=step2,
                               kernel_size=(3, 3), stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        # ------------
        self.layer3 = Xception_Block(in_filters=step2, out_filters=step3, strides=2, reps=2, start_with_relu=False,
                                     grow_first=True)
        self.layer4 = Xception_Block(in_filters=step3, out_filters=step4, strides=2, reps=2, start_with_relu=False,
                                     grow_first=True)
        self.layer5 = Xception_Block(in_filters=step4, out_filters=step5, strides=1, reps=2, start_with_relu=False,
                                     grow_first=True)

    def forward(self, inputs):
        inputs = self.relu1(self.bn1(self.conv1(inputs)))
        inputs = self.relu2(self.bn2(self.conv2(inputs)))
        inputs = self.layer3(inputs)
        inputs = self.layer4(inputs)
        inputs = self.layer5(inputs)
        return inputs


class Xception_middle_flow(nn.Module):
    def __init__(self, filters, rep):
        super(Xception_middle_flow, self).__init__()
        middle_path = []
        for i in range(rep):
            middle_path.append(Xception_Block(in_filters=filters, out_filters=filters,
                                              reps=3, start_with_relu=True, grow_first=False))
        self.net = nn.Sequential(*middle_path)

    def forward(self, x):
        return self.net(x)


class Xception_exit_flow(nn.Module):
    def __init__(self, in_filters, mid, out_filters):
        super(Xception_exit_flow, self).__init__()

        self.block12 = Xception_Block(in_filters=in_filters, out_filters=mid, reps=2, start_with_relu=True,
                                      grow_first=False)
        mid2 = (in_filters + out_filters) // 2
        self.conv3 = tl.SeparableConv2d(inplanes=mid, planes=mid2, kernel_size=3, stride=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(mid2)
        self.conv4 = tl.SeparableConv2d(inplanes=mid2, planes=out_filters, kernel_size=3, stride=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.block12(x)
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        return x


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        # entry_flow_conv
        self.entry = Xception_entry_flow(32, 64, 128, 256, 728)
        self.middle = Xception_middle_flow(728, 8)
        self.exit = Xception_exit_flow(728, 1024, 2048)
        # in

    def forward(self, x):
        x = self.entry(x)
        print(x.shape)
        x = self.middle(x)
        out = self.exit(x)
        return out
