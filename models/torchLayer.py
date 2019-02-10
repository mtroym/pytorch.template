import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu',
                 norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm != "None":
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != "None":
            return self.act(out)
        else:
            return out


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
        # print(inputs.shape)
        if pad_beg > inputs.shape[2] or pad_end > inputs.shape[3]:
            diff_0 = pad_beg - inputs.shape[2] + 1
            diff_1 = pad_end - inputs.shape[3] + 1
            inputs = F.pad(inputs, [inputs.shape[2] - 1, inputs.shape[3] - 1, inputs.shape[2] - 1, inputs.shape[3] - 1],
                           mode='reflect')
            pad_beg, pad_end = diff_0, diff_1
        padded_inputs = F.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end])
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
    def __init__(self, size):
        super(BilinearUpsampler, self).__init__()
        self.size = size

    def forward(self, input):
        return F.interpolate(input, self.size, mode='bilinear', align_corners=True)


if __name__ == '__main__':
    pass
