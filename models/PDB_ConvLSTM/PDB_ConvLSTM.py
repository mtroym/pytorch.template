# init
# PDC = ASPP + Xception bb
import numpy as np
import torch
import torch.nn as nn

from models.PDB_ConvLSTM.DB_ConvLSTM import DB_ConvLSTM_seq
from models.PDB_ConvLSTM.PDC import PDC
from models.backbones.Xception import Xception

BACKBONE = {'Xception': Xception}


class PDB_ConvLSTM(nn.Module):
    def __init__(self, inputsize, backbone, outstride, classes):
        super(PDB_ConvLSTM, self).__init__()
        if backbone == 'Xception':
            self.backbone = BACKBONE[backbone](outstride=outstride)
        self.x4size = (int(np.ceil(inputsize[0] / 4)), int(np.ceil(inputsize[1] / 4)))
        self.xoutsize = (int(np.ceil(inputsize[0] / outstride)), int(np.ceil(inputsize[1] / outstride)))
        print(self.xoutsize)
        self.PDC = PDC(2048, 256, size=self.xoutsize)
        self.inputsize = inputsize
        # self.catconv = tl.SeparableConv2d(256 * 5, 256, kernel_size=(1, 1), stride=1, bias=False, bn=False)
        self.x1DBconvLSTM = DB_ConvLSTM_seq(input_size=self.xoutsize, input_dim=256 * 5, hidden_dim=[512, 512],
                                            out_dim=256, kernel_size=(3, 3), dilation=1, bias=True, return_time=True,
                                            batch_first=True)
        self.x2DBconvLSTM = DB_ConvLSTM_seq(input_size=self.xoutsize, input_dim=256 * 5, hidden_dim=[512, 512],
                                            out_dim=256, kernel_size=(3, 3), dilation=2, bias=True, return_time=True,
                                            batch_first=True)

    def forward(self, input):
        seq_Len = x.size(1)
        out_temp = []
        for t in range(seq_Len):
            out_temp.append(self.backbone(x[:, t, :, :, :])[0])
        out = torch.stack(out_temp, dim=1)
        out = self.PDC(out)
        # out = self.catconv(out)
        out_1, hfw1, hbw1 = self.x1DBconvLSTM(out)
        out_2, hfw2, hbw2 = self.x2DBconvLSTM(out)
        out = torch.cat((out_1, out_2), dim=2)
        print(out.shape)
        return out


def createModel(opt):
    model = PDB_ConvLSTM(opt.inputSize, backbone='Xception', outstride=16, classes=opt.numClasses)
    if opt.GPU:
        model = model.cuda()
    return model


refers_to = "https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py"

if __name__ == '__main__':
    T = 5
    B = 3
    C = 1
    H = 20
    W = 20
    inputsize = (H, W)
    batch = 1
    net = PDB_ConvLSTM(inputsize, backbone='Xception', outstride=16, classes=4)
    x = torch.randn((B, T, C, H, W))
    print(net(x).shape)
