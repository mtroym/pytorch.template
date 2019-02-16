import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTM2dCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTM2dCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                            out_channels=4 * self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # cat input and hidden state.
        combined = torch.cat([input_tensor, h_cur], dim=1)
        # use a big convolutional layer to filter.
        combined_conv = self.conv(combined)
        # split each layer. cc for cat convolution.
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))


class ConvLSTM2d(nn.Module):
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
    @return_all_layers: bool

    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True):
        super(ConvLSTM2d, self).__init__()

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

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTM2dCell(input_size=(self.height, self.width),
                                            input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: optional
            4-D Tensor either of shape (b, C, k_s, k_s)

        Returns
        -------
        last_state_list, layer_output
        """

        if hidden_state[0] is not None and hidden_state[1] is not None:
            h_state_list, c_state_list = hidden_state
        else:
            h_state_list, c_state_list = self._init_hidden(batch_size=input_tensor.size(0))

        h_list = []
        c_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            hidden = (h_state_list[layer_idx], c_state_list[layer_idx])
            out, (h, c) = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                                    cur_state=hidden)
            h_list.append(h)
            c_list.append(c)

            cur_layer_input = h

        return out, (h_list, c_list)

    def _init_hidden(self, batch_size):
        h_list = []
        c_list = []
        for i in range(self.num_layers):
            h, c = self.cell_list[i].init_hidden(batch_size)
            h_list.append(h)
            c_list.append(c)
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
    L_seq = 2
    net = ConvLSTM2d(input_size=(h, w), input_dim=c, hidden_dim=[16, 32, 5], kernel_size=(3, 3), num_layers=3,
                     bias=True)
    x = torch.randn((B, c, h, w))
    print(x.shape)
    H = C = None
    for i in range(L_seq):
        y, (H, C) = net(x, (H, C))
        print(i, y.shape, len(H), len(C))
