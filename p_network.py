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

    def evaluate(self):
        return self.y_preds

    def _forward(self, cell, x, y, z, h):
        log_prob = []
        y_preds = []
        for t in range(self.T):
            log_prob_t, h_next, y_pred = cell(h, x[:,t], y[:,t], z[:,t])
            log_prob.append(log_prob_t)
            y_preds.append(y_pred)
            h = h_next
        sum = torch.sum(torch.stack(log_prob))
        self.y_preds = torch.stack(y_preds, dim=1)
        return sum, h

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
    def __init__(self, input_size, hidden_size, eps=1e-6):
        super(PCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = eps
        self.zcell = ZCell(input_size, hidden_size)
        self.ycell = YCell(input_size, hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, h, x, y_true, z_sample):
        # print(z_sample.size())
        prob_z = self.zcell(x, h, z_sample)
        # print('P network: x: {}, h: {}, z: {}'.format(x.size(), h_0[0].size(), z.size()))
        y_pred, h_next, prob_y = self.ycell(x, h, z_sample, y_true)
        # print('pcell: {}'.format(y))y
        # print('z: {}'.format(z))
        # print('y: {}'.format(y))
        # log_prob = torch.log(z_sample+self.eps) + torch.log(y[:, None]+self.eps)
        prob = prob_z + prob_y
        # print('log prob: {}'.format(log_prob))
        return prob, h_next, y_pred

class ZCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ZCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, 2 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(2 * hidden_size))

    def forward(self, x, hx, z_sample):
        h, c = hx

        wh = h.mm(self.weight_hh) + self.bias_hh
        wi = x.mm(self.weight_ih) + self.bias_ih
        i, f = torch.split(wh + wi, split_size_or_sections=self.hidden_size, dim=1)
        i = F.sigmoid(i)
        # print(i.size())
        f = F.sigmoid(f)
        # print(i.size())

        z = torch.cat((i, f),dim=1)
        # print('z: {}'.format(z.size()))

        prob = (z_sample * torch.log(z)) + ((1 - z_sample) * torch.log((1 - z)))
        return prob

class YCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(YCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.fcg = nn.Linear(input_size, 2*hidden_size)
        self.fco = nn.Linear(hidden_size, 2*hidden_size)

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, 2 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(2 * hidden_size))
        
        self.fcy1 = nn.Linear(hidden_size, 10*hidden_size)
        self.fcy2 = nn.Linear(10*hidden_size, input_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, hx, z, y_true):
        """ 
            x : [batch_size x input_size]
            z : [batch_size x num_gates]
            hx : ([batch_size x hidden_size], ...)
        """
        h_0, c_0 = hx
        # print(z.size())
        i = z[:,:self.hidden_size]
        f = z[:,self.hidden_size:]
        # print(i.size())

        # print("h_0: {}".format(h_0.size()))
        # print("weight: {}".format(self.weight_hh.size()))
        wh = h_0.mm(self.weight_hh) + self.bias_hh
        wi = x.mm(self.weight_ih) + self.bias_ih
        g, o = torch.split(wh + wi, split_size_or_sections=self.hidden_size, dim=1)
        
        g = F.tanh(g)
        o = F.sigmoid(o)
        c_1 = (f * c_0) + (i * g)
        # print('c_1: {}'.format(c_1))
        h_1 = o * F.tanh(c_1)
        hy = (h_1, c_1)
        # print('h_1: {}'.format(h_1))
        #TODO: add more layers here?
        y = F.relu(self.fcy1(h_1))
        y = F.relu(self.fcy2(y))
        y = F.sigmoid(y)
        prob = (y_true * torch.log(y)) + (1 - y_true) * torch.log((1 - y))
        # print('output: {}'.format(y))
        return y, hy, prob

        


        
