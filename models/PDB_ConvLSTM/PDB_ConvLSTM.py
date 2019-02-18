# init
# PDC = ASPP + Xception bb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

import models.LibTorchLayer as tl
from models.PDB_ConvLSTM.DB_ConvLSTM import DB_ConvLSTM_seq
from models.PDB_ConvLSTM.PDC import PDC
from models.backbones.Xception import Xception

BACKBONE = {'Xception': Xception}


class PDB_ConvLSTM(nn.Module):
    def __init__(self, input_dim, inputsize, backbone, outstride):
        super(PDB_ConvLSTM, self).__init__()
        self.input_dim = input_dim
        if backbone == 'Xception':
            self.backbone = BACKBONE[backbone](input_dim=self.input_dim, outstride=outstride)
        self.x4size = (int(np.ceil(inputsize[0] / 4)), int(np.ceil(inputsize[1] / 4)))
        self.xoutsize = (int(np.ceil(inputsize[0] / outstride)), int(np.ceil(inputsize[1] / outstride)))
        self.PDC = PDC(2048, 512, size=self.xoutsize)
        self.inputsize = inputsize
        self.catconv = tl.SeparableConv2d(512 * 4 + 2048, 32, kernel_size=(1, 1), stride=1, bias=False, bn=False,
                                          dilation=1)
        self.x1DBconvLSTM = DB_ConvLSTM_seq(input_size=self.xoutsize, input_dim=32, hidden_dim=[32, 32],
                                            kernel_size=(3, 3), dilation=1, bias=True, return_time=True,
                                            batch_first=True)
        self.x2DBconvLSTM = DB_ConvLSTM_seq(input_size=self.xoutsize, input_dim=32, hidden_dim=[32, 32],
                                            kernel_size=(3, 3), dilation=2, bias=True, return_time=True,
                                            batch_first=True)
        self.combine_conv = tl.SeparableConv2d(64, 1, kernel_size=(1, 1), stride=1, bias=False, bn=False, dilation=1)

    def forward(self, input):
        seq_Len = x.size(1)
        out_temp = []
        for t in range(seq_Len):
            out_temp.append(self.backbone(x[:, t, :, :, :])[0])
        out = torch.stack(out_temp, dim=1)

        print("Output shape after feature extraction:", out.shape)
        out = self.PDC(out)
        print("Output shape after PDC module:", out.shape)

        out_temp = []
        for t in range(seq_Len):
            out_temp.append(self.catconv(out[:, t, :, :, :]))
        out = torch.stack(out_temp, dim=1)
        print("Output shape after concatenate conv:", out.shape)

        out_1 = self.x1DBconvLSTM(out)
        out_2 = self.x2DBconvLSTM(out)
        out = torch.cat((out_1, out_2), dim=2)
        print("Output shape after DB-ConvLSTM:", out.shape)
        out_temp = []
        for t in range(seq_Len):
            out_temp.append(
                f.interpolate(torch.sigmoid(self.combine_conv(out[:, t, :, :, :])), self.inputsize, mode='bilinear',
                              align_corners=True))
        out = torch.stack(out_temp, dim=1)
        return out


def createModel(opt):
    model = PDB_ConvLSTM(opt.inputSize, backbone='Xception', outstride=8)
    if opt.GPU:
        model = model.cuda()
    return model


if __name__ == '__main__':
    T = 1
    B = 1
    C = 3
    H = 473
    W = 473
    inputsize = (H, W)
    batch = 1
    net = PDB_ConvLSTM(input_dim=C, inputsize=inputsize, backbone='Xception', outstride=8)
    x = torch.randn((B, T, C, H, W))
    print(net(x).shape)
