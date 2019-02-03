import torch
import torch.nn as nn

from models.backbones.Xception import Xception

BACKBONE = {'Xception': Xception}


class DeepLab(nn.Module):
    def __init__(self, backbone, outstride, classes):
        super(DeepLab, self).__init__()
        if backbone == 'Xception':
            self.backbone = BACKBONE[backbone](outstride=outstride)


def createModel(opt):
    # model = Net(opt)
    # if opt.GPU:
    #     model = model.cuda()
    # return model
    pass


refers_to = "https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py"

if __name__ == '__main__':
    import torch.nn.functional as F

    a = Xception(outstride=8)
    x = torch.randn((3, 3, 299, 299))

    b = a(x)
    print(b.shape)
    print(F.adaptive_avg_pool2d(b, (1, 1)).shape)
