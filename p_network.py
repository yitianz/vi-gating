import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

from inspect import signature

### P Network ###
# \prod p(y_t, h_t | h_t-1, x_t)

# First part: h_t-1, x_t -> z_t
# Second part: h_t-1, x_t, z_t -> y_t, h_t

class PNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        super(PNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        
        self.zs = []

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = PCell(layer_input_size, hidden_size)
            setattr(self, 'cell_{}'.format(layer), cell)

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        T = input_.size(0)
        output = []
        zs = []
        for t in range(T):
            y, hx_next, z = cell(input_[t], hx)
            output.append(y)
            zs.append(z)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx, zs

    def forward(self, input_, hx=None):
        T, batch_size, _ = input_.size()
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = [(hx, hx) for _ in range(self.num_layers)]
        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n), zs = PNetwork._forward_rnn(
                cell=cell, input_=input_, hx=hx[layer])
            new_hx.append((layer_h_n, layer_c_n))
            self.zs.append(zs)
        output = layer_output
        return output, new_hx

    def get_zs(self, layer):
        return self.zs[layer]

class PCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.train_c = TrainCell(input_size, hidden_size)
        self.output = OutputCell(input_size, hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h_0):
        x, h_0, z = self.train_c(x, h_0)
        # x, h_0, z = self.train(x)
        print('P network: x: {}, h: {}, z: {}'.format(x.size(), h_0[0].size(), z.size()))
        y, h_1 = self.output(x, h_0, z)
        return y, h_1, z

class TrainCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TrainCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: figure out right network
        self.fc1 = nn.Linear(input_size + 2*hidden_size, hidden_size)

    def forward(self, x, hx):
        #TODO: check dim
        print(x.size())
        h, c = hx
        input_ = torch.cat((x, h, c), dim=1)
        print(input_.size())

        layer = F.relu(self.fc1(input_))
        z = F.sigmoid(layer)
        print(z.size())
        return x, hx, z

class OutputCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OutputCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.fcg = nn.Linear(input_size, 2*hidden_size)
        self.fco = nn.Linear(hidden_size, 2*hidden_size)

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, 2 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(2 * hidden_size))
        
        #TODO: figure out what this network should be based on true y
        self.fc1 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, hx, z):
        h_0, c_0 = hx
        i, f = z.t()

        wh = h_0.mm(self.weight_hh) + self.bias_hh
        wi = x.mm(self.weight_ih) + self.bias_ih
        g, o = torch.split(wh + wi, split_size_or_sections=self.hidden_size, dim=1)
        # wh = self.fcg(h_0)
        # wi = self.fco(x)
        # g, o = torch.split(wh + wi, split_size_or_sections=self.hidden_size, dim=1)
        
        g = F.tanh(g)
        o = F.sigmoid(o)

        c_1 = (f * c_0) + (i * g)
        h_1 = o * F.tanh(c_1)

        hy = (c_1, h_1)

        y = F.relu(self.fc1(h_1))

        return y, hy

        


        
