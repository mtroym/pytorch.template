import torch
import torch.nn.functional as f
import torch.optim as optim
from torch import nn
from torch.autograd import Variable


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
            inputs = f.pad(inputs, [inputs.shape[2] - 1, inputs.shape[3] - 1, inputs.shape[2] - 1, inputs.shape[3] - 1],
                           mode='reflect')
            pad_beg, pad_end = diff_0, diff_1
        padded_inputs = f.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end])
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
        return f.interpolate(input, self.size, mode='bilinear', align_corners=True)


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class ConvLSTM1(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTM1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    b, c, h, w = 1, 3, 512, 512
    d = 1  # hidden state size
    lr = 5e-3  # learning rate
    T = 6  # sequence length
    max_epoch = 100  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTM(input_channels=3, hidden_channels=[4, 4], kernel_size=3, step=5, effective_step=[4])
    print(repr(model))

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, h, w))
    y = Variable(torch.randn(T, b, d, h, w))

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    print('Run for', max_epoch, 'iterations')
    optimizer.zero_grad()
    for epoch in range(0, max_epoch):
        loss = 0
        for t in range(0, T):
            state = model(x[t])
            print(state[0].shape)
            loss += loss_fn(state[0][t], y[t])

        print(' > Epoch {:2d} loss: {:.3f}, lr: {:.10f}'.format((epoch + 1), float(loss),
                                                                optimizer.param_groups[0]['lr']))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        optimizer.step()
        if epoch % 1 == 0:
            schedular.step()

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))
