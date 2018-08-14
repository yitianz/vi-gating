import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

### P Network ###
# \prod p(y_t, h_t | h_t-1, x_t)

# First part: h_t-1, x_t -> z_t
# Second part: h_t-1, x_t, z_t -> y_t, h_t

class PNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, T, num_layers=1, bias=True, dropout=0):
        super(PNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        
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

    def _forward(self, cell, x, y, z, h):
        log_prob = []
        for t in range(self.T):
            prob_t, h_next = cell(h, x[:,t], y[:,t], z[:,t])
            log_prob.append(torch.log(prob_t))
            h = h_next
        sum = torch.sum(torch.stack(log_prob))
        return sum, h

    # Pseudocode
    # def log_prob(self, x, y, z):
    #     log_prob = []
    #     h = torch.zeros(self.batch_size, self.hidden_size)
    #     h = (h, h)
    #     for t in range(self.T):
    #         h, log_prob_t = p_cell(h, x[:,t], y[:,t], z[:, t])
    #         log_prob.append(log_prob_t)
    #     return torch.sum(torch.stack(log_prob))

    def forward(self, x, y, z):
        batch_size, T, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)
        h = [(h, h) for _ in range(self.num_layers)]

        new_hx = []
        layer_output = None
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = self._forward(cell=cell, x=x, y=y, z=z, h=h[layer])
            new_hx.append((layer_h_n, layer_c_n))
        output = layer_output
        return output, new_hx

# log p = \sum_t log p (z_t | h_t-1, x_t) + log p (y_t | h_t-1, x_t, z_t)
# h_t = OutputCell(h_t-1, x_t, z_t)

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

    def forward(self, h, x, y, z_sample):
        x, h, z = self.train_c(x, h)
        # print('P network: x: {}, h: {}, z: {}'.format(x.size(), h_0[0].size(), z.size()))
        # TODO: check that should be using sampled z in this part
        y, h_next = self.output(x, h, z_sample)
        print('z: {}'.format(z))
        print('y: {}'.format(y))
        # TODO: is this returning the right thing?
        log_prob = torch.log(z) + torch.log(y)
        return log_prob, h_next

class TrainCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TrainCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: figure out right network
        self.fc1 = nn.Linear(input_size + 2*hidden_size, hidden_size)

    def forward(self, x, hx):
        h, c = hx
        input_ = torch.cat((x, h, c), dim=1)

        layer = F.relu(self.fc1(input_))
        z = F.sigmoid(layer)
        return x, hx, z

class OutputWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        super(OutputWrapper, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        for t in range(T):
            cell = OutputCell(input_size, hidden_size)
            setattr(self, 'cell_{}'.format(t), cell)

    def get_cell(self, t):
        return getattr(self, 'cell_{}'.format(t))

    def forward(self, x, z):
        h = Variable(torch.zeros(1, self.hidden_size))
        h = (h, h)
        output = []
        for t in range(self.T):
            cell = self.get_cell(t)
            y, h_1 = cell(x[t], h, z[t])
            output.append(y)
            h = h_1
        output = torch.stack(output, 0)
        return output

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
        
        self.fc1 = nn.Linear(hidden_size, input_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, hx, z):
        h_0, c_0 = hx
        i = z[:,0]
        f = z[:,1]

        wh = h_0.mm(self.weight_hh) + self.bias_hh
        wi = x.mm(self.weight_ih) + self.bias_ih
        g, o = torch.split(wh + wi, split_size_or_sections=self.hidden_size, dim=1)
        
        g = F.tanh(g)
        o = F.sigmoid(o)

        c_1 = (f * c_0) + (i * g)
        h_1 = o * F.tanh(c_1)

        hy = (h_1, c_1)
        print('h_1: {}'.format(h_1))
        #TODO: figure out what this network should be based on true y
        y = F.relu(self.fc1(h_1))

        return y, hy

        


        
