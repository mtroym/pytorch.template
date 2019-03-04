from models.backbones import Xception

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'xception':
        return Xception.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError