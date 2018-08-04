import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from p_network import OutputCell

### Q Network ###
# \prod q(z_t | x_t, y_t)
# 1. x, y -> b
# 2. b -> z
# 3. z, x -> y

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.brnn = nn.LSTM(2 * input_size, hidden_size, bidirectional=True)

        self.zs = []

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = QCell(layer_input_size, hidden_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, hx, b):
        T = input_.size(0)
        output = []
        zs = []
        for t in range(T):
            y, hx_next, z = cell(input_[t], hx, b[t].unsqueeze(0))
            output.append(y)
            zs.append(z)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx, zs

    def forward(self, x, y, hx=None):
        #TODO: check correctness of getting BRNN outputs
        input_ = torch.cat((x, y), dim=2)
        output, (h_n, c_n) = self.brnn(input_)
        # forward = output[-1, :, :self.hidden_size]
        # reverse = output[0, :, self.hidden_size:]
        forward = output[:, 0, :self.hidden_size]
        reverse = output[:, 0, self.hidden_size:]
        context = torch.cat((forward, reverse), dim=1)

        # run through layers and cells
        T, batch_size, _ = input_.size()
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = [(hx, hx)]
        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = InferenceCell(self.input_size, self.hidden_size)
            layer_output, (layer_h_n, layer_c_n), zs = QNetwork._forward_rnn(
                cell=cell, input_=x, hx=hx[layer], b=context)
            new_hx.append((layer_h_n, layer_c_n))
            self.zs.append(zs)
        output = layer_output
        return output, new_hx

    def sample(self, layer, step):
        # sample for all time steps at once
        z = torch.Tensor(self.zs[layer][step])
        ber = torch.distributions.Bernoulli(z)
        samples = ber.sample()
        return samples

    def get_zs(self, layer):
        return self.zs[layer]


class QCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.inference = InferenceCell(input_size, hidden_size)
        # same as PCell
        self.output = OutputCell(input_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h_0, b):
        z = self.inference(b)
        print('Q network: x: {}, h: {}, z: {}'.format(x.size(), h_0[0].size(), z.size()))
        y, h_1 = self.output(x, h_0, z)
        return y, h_1, z
    

#QInferenceWrapper(x, y,) -> b -> z
class QInferenceWrapper(nn.Module):
    # TODO: add more arguments per QNetwork
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(QInferenceWrapper, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.brnn = nn.LSTM(2 * input_size, hidden_size, bidirectional=True)
        self.num_layers = num_layers

        for layer in range(self.num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = InferenceCell(layer_input_size, hidden_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, b):
        T = input_.size(0)
        output = []
        zs = []
        for t in range(T):
            z = cell(b[t].unsqueeze(0))
            output.append(z)
        output = torch.stack(output, 0)
        return output

    def forward(self, x, y, hx=None):
        input_ = torch.cat((x, y), dim=2)
        output, (h_n, c_n) = self.brnn(input_)
        # forward = output[-1, :, :self.hidden_size]
        # reverse = output[0, :, self.hidden_size:]
        forward = output[:, 0, :self.hidden_size]
        reverse = output[:, 0, self.hidden_size:]
        context = torch.cat((forward, reverse), dim=1)

        # run through layers and cells
        T, batch_size, _ = input_.size()
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = [(hx, hx) for _ in range(self.num_layers)]
        layer_output = None
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = self.get_cell(layer)
            layer_output = QInferenceWrapper._forward_rnn(
                cell=cell, input_=x, b=context)
        output = layer_output
        return output

class InferenceCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InferenceCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        # TODO: add more layers here

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, b):
        layer = F.relu(self.fc1(b))
        z = F.sigmoid(layer)
        return z
        
