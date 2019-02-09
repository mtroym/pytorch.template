import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.deeplab.ASPP import ASPP
from models.backbones.Xception import Xception

BACKBONE = {'Xception': Xception}


class DeepLab(nn.Module):
    def __init__(self, inputsize, backbone, outstride, classes):
        super(DeepLab, self).__init__()
        if backbone == 'Xception':
            self.backbone = BACKBONE[backbone](outstride=outstride)
        self.x4size = (int(np.ceil(inputsize[0] / np.sqrt(outstride))), int(np.ceil(inputsize[1] / np.sqrt(outstride))))
        self.x16size = (int(np.ceil(inputsize[0] / outstride)), int(np.ceil(inputsize[1] / outstride)))
        self.ASPP = ASPP(2048, 256, size=self.x16size)
        self.inputsize = inputsize
        self.x4conv = nn.Conv2d(128, 256, kernel_size=(1, 1), padding=0, stride=1, bias=False)
        self.decoder_conv = nn.Conv2d(in_channels=256*2, out_channels=classes, kernel_size=(3, 3), padding=1)

    def forward(self, input):
        input, low_level_feat = self.backbone(input)
        input = self.ASPP(input)
        decode_aspp = F.interpolate(input, self.x4size, mode='bilinear', align_corners=True)
        decode_feat = self.x4conv(low_level_feat)
        print(decode_aspp.shape)
        decode_in = torch.cat((decode_aspp, decode_feat), 1)
        out = self.decoder_conv(decode_in)
        out = F.interpolate(out, self.inputsize, mode='bilinear', align_corners=True)
        return out


def createModel(opt):
    model = DeepLab(opt.inputSize, backbone='Xception', outstride=16, classes=21)
    if opt.GPU:
        model = model.cuda()
    return model


refers_to = "https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py"

if __name__ == '__main__':
    a = DeepLab(inputsize=(299, 299), backbone='Xception', outstride=16, classes=21)
    x = torch.randn((1, 3, 299, 299))
    print(a(x).shape)
    # print(299.0 / 2 / 2 / 2 / 2)
