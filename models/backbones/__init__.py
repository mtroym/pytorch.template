from models.backbones import Xception, Resnet


def build_backbone(input_dim, backbone, output_stride, BatchNorm):
    if backbone == 'Xception':
        return Xception.AlignedXception(input_dim, output_stride, BatchNorm, pretrained=False)
    elif backbone == 'Resnet':
        return Resnet.ResNet101(input_dim, output_stride, BatchNorm, pretrained=False)
    else:
        raise NotImplementedError
