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
    def __init__(self, input_size, hidden_size, T, num_layers=1, bias=True, dropout=0, eps=1e-6):
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.eps=eps

        self.brnn = nn.LSTM(2 * input_size, hidden_size, bidirectional=True, batch_first=True)

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

# log_q = \sum_t log(z_t | b_t, h_t-1)
# b_t = BRNN(x, y)
    def _forward(self, cell, x, h, b, z_sample=None):
        log_prob = []
        for t in range(self.T):
            prob_t, h_next = cell(x[:,t], h, b[:,t].unsqueeze(0), z_sample)
            log_prob.append(torch.log(prob_t+self.eps))
            h = h_next
        sum = torch.sum(torch.stack(log_prob))
        return sum, h

    def forward(self, x, y, z):
        input_ = torch.cat((x, y), dim=2)
        output, (h_n, c_n) = self.brnn(input_)
        # TODO: does this work?
        context = output
        # forward = output[:, 0, :self.hidden_size]
        # reverse = output[:, 0, self.hidden_size:]
        # context = torch.cat((forward, reverse), dim=1)
        # run through layers and cells
        batch_size, T, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)
        h = [(h, h) for _ in range(self.num_layers)]

        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = QCell(self.input_size, self.hidden_size)
            layer_output, (layer_h_n, layer_c_n)= self._forward(cell=cell, x=x, h=h[layer], b=context, z_sample=z)
            new_hx.append((layer_h_n, layer_c_n))
        output = layer_output
        return output, new_hx

    def _sample(self, cell, x, h, b, z_sample=None):
        zs = []
        for t in range(self.T):
            z, _ = cell(x[:,t], h, b[:,t].unsqueeze(0), z_sample)
            zs.append(z)
        return torch.stack(zs, dim=2).squeeze(0)

    def sample(self, x, y, layer):
        input_ = torch.cat((x, y), dim=2)
        output, (h_n, c_n) = self.brnn(input_)
        # TODO: does this work?
        context = output

        batch_size, T, _ = x.size()
        h = torch.zeros(self.num_layers, self.hidden_size)
        h = [(h, h) for _ in range(batch_size)]

        layer_output = None
        zs = []
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = QCell(self.input_size, self.hidden_size)
            layer_output = self._sample(cell=cell, x=x, h=h[layer], b=context)
            zs.append(layer_output)
        samples = []
        for layer in range(self.num_layers):
            ber = torch.distributions.Bernoulli(zs[layer])
            sample = ber.sample()
            samples.append(sample)
        return torch.stack(samples).squeeze(0)
        # z = torch.exp(torch.Tensor(zs[layer]))
        # ber = torch.distributions.Bernoulli(z)
        # samples = ber.sample()
        # return samples

class QCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-6):
        super(QCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = eps

        self.output = OutputCell(input_size, hidden_size)

        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        # TODO: add more layers here

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h, b, z_sample=None):
        layer = F.relu(self.fc1(b))
        z = F.sigmoid(layer)
        if z_sample is not None:
            _, h_next = self.output(x, h, z_sample)
            return z, h_next
        else:
            return z, None
        
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
        
