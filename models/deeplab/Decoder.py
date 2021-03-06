import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, with_height=False):
        super(Decoder, self).__init__()
        if backbone == 'Resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'Xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        if with_height:
            low_level_inplanes += 1
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        last_conv_inchannel = 304
        self.last_conv = nn.Sequential(nn.Conv2d(last_conv_inchannel, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat, h=None):
        if h is not None:
            hs = F.interpolate(h, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
            low_level_feat = torch.cat((low_level_feat, hs), dim=1)
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, with_height=False):
    return Decoder(num_classes, backbone, BatchNorm, with_height)


if __name__ == '__main__':
    decoder = build_decoder(5, 'Resnet', nn.BatchNorm2d)
    print(decoder)