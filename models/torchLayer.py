import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, bn=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=0,
                                   dilation=dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=bias)
        self.bn = bn
        self.depthwise_bn = nn.BatchNorm2d(inplanes) if self.bn else None
        self.pointwise_bn = nn.BatchNorm2d(planes) if self.bn else None

    @staticmethod
    def fixed_padding(inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end], mode='reflect')
        return padded_inputs

    def forward(self, inputs):
        inputs = self.fixed_padding(inputs, self.depthwise.kernel_size[0],
                                    dilation=self.depthwise.dilation[0])
        inputs = self.depthwise(inputs)
        if self.bn:
            inputs = self.depthwise_bn(inputs)
        inputs = self.pointwise(inputs)
        if self.bn:
            inputs = self.pointwise_bn(inputs)
        return inputs

class BilinearUpsampler(nn.Module):
    def __init__(self):
        super(BilinearUpsampler, self).__init__()



if __name__ == '__main__':
    pass
