from models.backbones import Xception, Resnet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'Xception':
        return Xception.AlignedXception(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'Resnet':
        return Resnet.ResNet101(output_stride, BatchNorm, pretrained=False)
    else:
        raise NotImplementedError
