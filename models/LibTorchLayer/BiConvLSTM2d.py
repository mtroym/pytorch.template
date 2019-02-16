import copy

import torch
import torch.nn as nn

import models.LibTorchLayer as tl


class BiConvLSTM2d(nn.Module):
    """
    @input_size: (int, int)
        Height and width of input tensor as (height, width).
    @input_dim: int
        Number of channels of input tensor.
    @hidden_dim: int
        Number of channels of hidden state.
    @kernel_size: (int, int)
        Size of the convolutional kernel.
    @bias: bool
        Whether or not to add the bias.
    @batch_first: bool
        Whether the data is (B, T, C, H, W) or (T, B, C, H, W)
    @bias: bool
        Whether there is a bias term
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(BiConvLSTM2d, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        input_dim_list = [self.input_dim] + self.hidden_dim
        cell_list_fw = [tl.ConvLSTM2dCell(input_size=(self.height, self.width),
                                          input_dim=input_dim_list[i],
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias) for i in range(self.num_layers)]
        cell_list_bw = copy.deepcopy(cell_list_fw)
        self.combined_conv = tl.SeparableConv2d(inplanes=self.hidden_dim[-1] * 2,
                                                planes=self.hidden_dim[-1],
                                                kernel_size=self.kernel_size[-1],
                                                bias=self.bias)
        self.cell_list_fw = nn.ModuleList(cell_list_fw)
        self.cell_list_bw = nn.ModuleList(cell_list_bw)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (2 * b, c, h, w)
        hidden_state: optional
            4-D Tensor either of shape (b, C, k_s, k_s)

        Returns:         output, (h_list, c_list)
        -------
        output:
            output tensor. (b, OC, h, w)
        h_list:
            list of hidden Tensor of len num_layers
        c_list:
            list of hidden Tensor of len num_layers
        """
        if hidden_state[0] is not None and hidden_state[1] is not None:
            h_state_list, c_state_list = hidden_state
        else:
            h_state_list, c_state_list = self._init_hidden(batch_size=int(input_tensor.size(0)/2))

        layer_output_list_fw = []
        layer_output_list_bw = []

        h_list = []
        c_list = []

        cur_layer_input_fw, cur_layer_input_bw = input_tensor.chunk(2, dim=0)
        for layer_idx in range(self.num_layers):
            cur_layer_input_fw, (h_fw, c_fw) = self.cell_list_fw[layer_idx](input_tensor=cur_layer_input_fw,
                                                                            cur_state=h_state_list[layer_idx])
            cur_layer_input_bw, (h_bw, c_bw) = self.cell_list_bw[layer_idx](input_tensor=cur_layer_input_bw,
                                                                            cur_state=c_state_list[layer_idx])
            layer_output_fw = h_fw
            layer_output_bw = h_bw

            layer_output_list_fw.append(layer_output_fw)
            layer_output_list_bw.append(layer_output_bw)
            h_list.append((h_fw, h_bw))
            c_list.append((c_fw, c_bw))

        combined = torch.cat((h_fw, h_bw), dim=1)
        output = self.combined_conv(combined)
        return output, (h_list, c_list)

    def _init_hidden(self, batch_size):
        h_list = []
        c_list = []
        for i in range(self.num_layers):
            h_fw, c_fw = self.cell_list_fw[i].init_hidden(batch_size)
            h_bw, c_bw = self.cell_list_bw[i].init_hidden(batch_size)
            h_list.append((h_fw, h_bw))
            c_list.append((c_fw, c_bw))
        return (h_list, c_list)

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    T = 10
    B = 16
    c = 3
    h = 25
    w = 25
    L_seq = 60
    net = BiConvLSTM2d(input_size=(h, w), input_dim=c, hidden_dim=[16, 32, 5], kernel_size=(3, 3), num_layers=3, bias=True)
    x = torch.randn((B, c, h, w))
    x_r = torch.randn((B, c, h, w))
    X = torch.cat((x, x_r), dim=0)
    print(X.shape)
    H = C = None
    for i in range(L_seq):
        y, (H, C) = net(X, (H, C))
        print(i, y.shape, len(H), len(C))

    # net = nn.LSTM(input_size=h, hidden_size=10)
