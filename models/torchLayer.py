import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end])
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
# pad_total = kernel_size_effective - 1
# pad_beg = pad_total // 2
# pad_end = pad_total - pad_beg
# x = ZeroPadding2D((pad_beg, pad_end))(x)
def make_pair(x):
    if isinstance(x, (list, tuple)):
        if len(x) != 2:
            return (x[0],) * 2
        else:
            return x
    else:
        return (x,)*2


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding="VALID",
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_pair(kernel_size)
        self.stride = make_pair(stride)
        self.padding = padding
        self.dilation = make_pair(dilation)
        self.groups = groups
        self.weight = Parameter(torch.Tensor(
            self.out_channels, self.in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def cal_pad(self):
        pass

    def forward(self, inputs):
        padd = cal_pad_if_same(kernel_size=self.kernel_size, dilation=self.dilation)
        padded_inputs = F.pad(inputs, [padd[0]//2, padd[1]//2, padd[0]//2, padd[1]//2])
        return F.conv2d(padded_inputs, self.weight, self.bias, self.stride, padding=0)

def cal_pad_if_same(kernel_size, dilation):
    k = kernel_size
    d = dilation
    p = [0, 0]
    for j in range(2):
        p[j] = k[j] + (k[j] - 1)*(d[j] - 1) - 1
    return tuple(p)

if __name__ == '__main__':
    print(cal_pad_if_same(make_pair(3), make_pair(1)))
    conv = Conv2d(in_channels=3, out_channels=32, kernel_size=3,
           stride=2, padding='SAME', dilation=2)
    x = torch.randn((3, 3, 30, 30))
    print(conv(x).shape)