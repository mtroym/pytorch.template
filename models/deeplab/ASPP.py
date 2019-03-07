import torch
import torch.nn as nn
import models.LibTorchLayer as tl
import torch.nn.functional as F


class _ASPP(nn.Module):
    def __init__(self, inplane, outplane, rates=(1, 6, 12, 18), size=None):
        super(_ASPP, self).__init__()
        self.size = size
        self.at_pool1x1 = tl.SeparableConv2d(inplane, outplane, kernel_size=(1, 1), dilation=rates[0], bn=True)
        self.at_pool3x3_1 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=rates[1], bn=True)
        self.at_pool3x3_2 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=rates[2], bn=True)
        self.at_pool3x3_3 = tl.SeparableConv2d(inplane, outplane, kernel_size=(3, 3), dilation=rates[3], bn=True)
        self.concate_project_conv = nn.Conv2d(outplane * 5, outplane, kernel_size=(1, 1))
        self.feat_conv = nn.Conv2d(inplane, outplane, kernel_size=(1, 1), bias=False)
        self.concate_project_bn = nn.BatchNorm2d(outplane)
        self.concate_project_drop = nn.Dropout2d(0.1)
        self.at_poolall = nn.AvgPool2d(kernel_size=self.size)

    def forward(self, x):
        feat = self.at_poolall(x)
        feat = F.interpolate(self.feat_conv(feat), self.size, mode='bilinear', align_corners=True)
        x1conv = self.at_pool1x1(x)
        x3_conv_1 = self.at_pool3x3_1(x)
        x3_conv_2 = self.at_pool3x3_2(x)
        x3_conv_3 = self.at_pool3x3_3(x)
        at_all = torch.cat((x1conv, x3_conv_1, x3_conv_2, x3_conv_3), 1)
        at_all = F.interpolate(at_all, self.size, mode='bilinear', align_corners=True)
        all = torch.cat((at_all, feat), 1)
        out = self.concate_project_drop(F.relu(self.concate_project_bn(self.concate_project_conv(all))))
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else: # Resnet
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)






if __name__ == '__main__':
    ASPP = build_aspp('Xception', 8, nn.BatchNorm2d)
    print(ASPP)
    # net = ASPP(2048, 256, size=(int(299 / 8), int(299 / 8)))
    # x = torch.randn((1, 2048, int(299 / 8), int(299 / 8)))
    # print(net(x).shape)


