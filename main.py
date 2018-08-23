import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

from graphviz import Digraph
import re

import p_network
import q_network

from p_network import PNetwork
from q_network import QNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Config ###
log_interval = 20
num_epochs = 1000
num_data = 10
batch_size = 5
num_batches = int(num_data / batch_size)
seq_length = 20
num_layers = 1
input_size = 1
hidden_size = 2
lr = 1e-2

# Set the random seed manually for reproducibility.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         message("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed(args.seed)

def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.grad_fn)
    return dot

### Data ###
# x: {0,1} # y:{0,1} 
# y = xor(x) -> 0 until the 1st x, then flip to 1
def generate_data():
    """ xs and ys : [batch_size x seq_length x input_size] """
    xs = []
    ys = []
    for batch in range(num_batches):
        x = [np.random.randint(0, 2) for _ in range(seq_length)]
        xs.append(x)
        index = x.index(1)
        y = [0 for _ in range(index)] + [1 for _ in range(len(x) - index)]
        ys.append(y)
    if num_batches > batch_size:
        remainder = num_batches % batch_size
        while remainder != 0:
            xs.append([0 for _ in range(seq_length)])
            ys.append([0 for _ in range(seq_length)])
            remainder -= 1
    xs = torch.Tensor(xs).unsqueeze(2).to(device)
    ys = torch.Tensor(ys).unsqueeze(2).to(device)
    print('x: {}'.format(xs[:, :, 0]))
    print('y: {}'.format(ys[:, :, 0]))
    return xs, ys

q = QNetwork(input_size, hidden_size, seq_length).to(device)
p = PNetwork(input_size, hidden_size, seq_length).to(device)
params = list(q.parameters()) + list(p.parameters())
optimizer = torch.optim.Adagrad(params, lr=lr)


def train(xs, ys):
    q.train()
    p.train()
    train_loss = 0
    for epoch in range(num_epochs):
        for b in range(0, num_batches, batch_size):
            
            z = q.sample(xs[b:b+batch_size], ys[b:b+batch_size], 0)
            z = z.detach()
            # print(z.size())
            # print("main: {}".format(z.size()))
            """ z : [batch_size x seq_length x (num_gates * hidden_size)] """
            # log p(z, y | x)
            log_p, _ = p(xs[b:b+batch_size], ys[b:b+batch_size], z)
            # log q(z | x, y)
            log_q, _ = q(xs[b:b+batch_size], ys[b:b+batch_size], z)
            
            f = make_dot(log_p)
            # f.view()

            g = make_dot(log_q)
            # g.view()
            # score function trick
            opt_loss = -( (log_q * (log_p - log_q).detach()) + (log_p - log_q) )
            obj_loss = -(log_p - log_q)
            if epoch % log_interval == 0:
                print('objective loss epoch {}: {}'.format(epoch, obj_loss.item()))
                # print('optimization loss epoch {}: {}'.format(epoch, opt_loss.item()))
            optimizer.zero_grad()
            opt_loss.backward()
            optimizer.step()
            train_loss += opt_loss.item()

def test(xs, ys):
    q.eval()
    p.eval()
    print(p)
    test_loss = 0
    y_loss = 0
    for epoch in range(num_epochs):
        for b in range(0, num_batches, batch_size):
            z = q.sample(xs[b:b+batch_size], ys[b:b+batch_size], 0)
            # log p(z, y | x)
            log_p, _ = p(xs[b:b+batch_size], ys[b:b+batch_size], z)
            # log q(z | x, y)
            log_q, _ = q(xs[b:b+batch_size], ys[b:b+batch_size], z)

           
            # y_loss += nn.MSELoss()(y_preds, ys[b:b+batch_size])

            opt_loss = -( (log_q * (log_p - log_q).detach()) + (log_p - log_q) )
            obj_loss = -(log_p - log_q)
            test_loss += opt_loss.item()
            if epoch % log_interval == 0:
                print('objective loss epoch {}: {}'.format(epoch, obj_loss.item()))
                print('x: {}'.format(xs[b:b+batch_size, :, 0]))
                y_preds = p.evaluate()
                print('y: {}'.format(y_preds[:,:,0]))
                squared_loss = lambda a, b : ((a - b) ** 2)
                y_loss = torch.mean(squared_loss(y_preds, ys[b:b+batch_size]))
                print('y loss: {}'.format(y_loss.item()))

xs_train, ys_train = generate_data()
train(xs_train, ys_train)

xs_test, ys_test = generate_data()
test(xs_test, ys_test)









