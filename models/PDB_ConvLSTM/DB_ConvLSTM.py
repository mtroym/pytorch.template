import torch
import torch.nn as nn

import models.LibTorchLayer as tl


class DB_ConvLSTM_seq(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 dilation=1, bias=True, return_time=True, batch_first=True):
        super(DB_ConvLSTM_seq, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, 2)
        hidden_dim = self._extend_for_multilayer(hidden_dim, 2)
        if not len(kernel_size) == len(hidden_dim) == 2:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = 2
        self.bias = bias
        self.batch_first = batch_first
        self.return_time = return_time
        self.dilation = dilation
        self.cell_fw = tl.ConvLSTM2dCell(input_size=(self.height, self.width),
                                         input_dim=self.input_dim,
                                         hidden_dim=self.hidden_dim[0],
                                         kernel_size=self.kernel_size[0],
                                         dilation=self.dilation,
                                         bias=self.bias)
        self.cell_bw = tl.ConvLSTM2dCell(input_size=(self.height, self.width),
                                         input_dim=self.hidden_dim[0],
                                         hidden_dim=self.hidden_dim[1],
                                         dilation=self.dilation,
                                         kernel_size=self.kernel_size[1],
                                         bias=self.bias)
        self.combined_conv = tl.SeparableConv2d(inplanes=self.hidden_dim[0] + self.hidden_dim[1],
                                                planes=self.hidden_dim[-1],
                                                kernel_size=(3, 3),
                                                dilation=1,
                                                bias=self.bias)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: optional
            4-D Tensor either of shape (b, C, k_s, k_s)

        Returns: output, (h_list, c_list)
        -------
        output:
            output tensor. (b, OC, h, w)
        h_list:
            2d list of hidden Tensor of len num_layers of len time_seq
        c_list:
            2d list of hidden Tensor of len num_layers of len time_seq
        """
        if not self.batch_first:
            input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_fw, hidden_bw = self._init_hidden(batch_size=input_tensor.size(0))

        seq_len = input_tensor.size(1)
        out_fw_list = []
        out_bw_list = []
        # fw pass
        for t in range(seq_len):
            out_fw, hidden_fw = self.cell_fw(input_tensor=input_tensor[:, t, :, :, :],
                                             cur_state=hidden_fw)
            out_fw_list.append(out_fw)

        fw_layer_out = torch.stack(out_fw_list, dim=1)
        # bw pass
        for t in range(seq_len):
            out_bw, hidden_bw = self.cell_bw(input_tensor=fw_layer_out[:, seq_len - 1 - t, :, :, :],
                                             cur_state=hidden_bw)
            out_bw_list.append(out_bw)
        bw_layer_out = torch.stack(out_bw_list, dim=1)
        cat_out = torch.cat((fw_layer_out, bw_layer_out), dim=2)
        temp_out = []
        for t in range(seq_len):
            temp_out.append(torch.tanh(self.combined_conv(cat_out[:, t, :, :, :])))
        out = torch.stack(temp_out, dim=1)
        return out

    def _init_hidden(self, batch_size):
        init_states = ((self.cell_fw.init_hidden(batch_size), self.cell_bw.init_hidden(batch_size)))
        return init_states

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
    T = 3
    B = 5
    c = 2048
    h = 60
    w = 60
    net = DB_ConvLSTM_seq(input_size=(h, w), input_dim=c, hidden_dim=[16, 7], kernel_size=(3, 3), bias=True,
                          batch_first=True, dilation=2)
    x = torch.randn((B, T, c, h, w))
    print(type(x))
    y = net(x)
    print(y.shape)

    # net = nn.LSTM(input_size=h, hidden_size=10)
