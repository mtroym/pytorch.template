import torch.nn as nn

from models.deeplabz.deeplabz import DeepLabz


class DeepLabz3d(nn.Module):
    def __init__(self, input_dim=3, backbone='Resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabz3d, self).__init__()

        self.model_xy = DeepLabz(input_dim=input_dim, backbone=backbone, output_stride=output_stride,
                                 num_classes=num_classes, sync_bn=sync_bn, freeze_bn=freeze_bn)
        self.model_xz = DeepLabz(input_dim=input_dim, backbone=backbone, output_stride=output_stride,
                                 num_classes=num_classes, sync_bn=sync_bn, freeze_bn=freeze_bn)
        self.model_yz = DeepLabz(input_dim=input_dim, backbone=backbone, output_stride=output_stride,
                                 num_classes=num_classes, sync_bn=sync_bn, freeze_bn=freeze_bn)



    def forward(self, *input):
        pass


def createModel(opt):
    model = DeepLabz3d(input_dim=opt.input_dim, backbone=opt.backbone, output_stride=opt.outStride,
                       num_classes=opt.numClasses, sync_bn=False,
                       freeze_bn=False)
    if opt.GPU:
        model = model.cuda()
    return model
