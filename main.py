import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

import p_network
import q_network

from p_network import OutputWrapper
from p_network import OutputCell
from p_network import TrainCell


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data ###
# x: {0,1} # y:{0,1} 
# y = xor(x) -> 0 until the 1st x, then flip to 1
length = 10
input_size = 1
hidden_size = 2
lr = 1e-3

x = [np.random.randint(0, 2) for _ in range(length)]
zeros = np.zeros(length)
index = x.index(1)
y = [0 for _ in range(index)] + [1 for _ in range(len(x) - index)]
xs = torch.Tensor(x).unsqueeze(1).unsqueeze(2).to(device)
ys = torch.Tensor(y).unsqueeze(1).unsqueeze(2).to(device)
print('x: {}'.format(x))
print('y: {}'.format(y))

### Helper functions ###

def loss_function_z(z, p_output, q_output):
    p_term = z*torch.log(p_output) + (1-z)*torch.log(p_output)
    q_term = z*torch.log(q_output) + (1-z)*torch.log(q_output)
    sub = p_term - q_term
    return sub.sum()

def loss_function_y(y_pred, y_true):
    return nn.MSELoss()(y_pred, y_true)

def loss_score_function(p, q):
    return torch.log(p / q) * torch.log(q)

def loss_function_q(z_pred, z_true):
    return nn.MSELoss()(z_pred, z_true)

def sample_z(q_output, step):
    z = torch.Tensor(q_output[step])
    ber = torch.distributions.Bernoulli(z)
    sample = ber.sample()
    return sample

# p_model = p_network.PNetwork(input_size, hidden_size).to(device)
# q_model = q_network.QNetwork(input_size, hidden_size).to(device)
# parameters = list(p_model.parameters()) + list(q_model.parameters())
# optimizer = torch.optim.SGD(parameters, lr=lr)
# for each datapoint (x, y):
#    sample z | x, y ~ q_\phi
#    take gradient step for \theta and \phi of loss log (p_\theta(y, z | x) / q_\phi(z | x, y) )
# p_model.train()
# q_model.train()

# q_output, _ = q_model(x, y)
# p_output, _ = p_model(x)

# train_loss = 0

# for i in range(length):
#     optimizer.zero_grad()
#     z = q_model.sample(0, i)
#     p_i = p_output[i]
#     q_i = q_output[i]
#     loss = torch.log(p_i / q_i) * torch.log(q_i)
#     loss.backward(retain_graph=True)
#     print(loss.item())
#     train_loss += loss.item()
#     optimizer.step()

### Training ###

# 1. run Q inference cell through all time steps to get z distribution

num_iters = 10
q_inference = q_network.QInferenceWrapper(input_size, hidden_size).to(device)
q_optim = torch.optim.SGD(q_inference.parameters(), lr=lr)
q_inference.train()
q_loss = 0

for i in range(num_iters):
    q_optim.zero_grad()
    q_inference_output = q_inference(xs, ys)

    #TODO: how to get true values of z?
    eps = torch.rand(length, 1, hidden_size)
    q_inference_true = q_inference_output + eps
    q_inference_true.detach_()

    loss = loss_function_q(q_inference_output, q_inference_true)
    loss.backward()
    q_loss += loss.item()
    q_optim.step()

# 2. use sampled z to train P train cell using loss function that minimizes difference  w/ sampled z

zs = [sample_z(q_inference_output, i) for i in range(length)]

p_train = p_network.TrainCell(input_size, hidden_size).to(device)
p_optim = torch.optim.SGD(p_train.parameters(), lr=lr)
p_train.train()

p_train_loss = 0
h = Variable(torch.zeros(1, hidden_size))
h = (h, h)

for i in range(length):
    p_optim.zero_grad()
    output = OutputCell(input_size, hidden_size)
    y, h_1 = output(xs[i], h, zs[i])

    _, _, p_i = p_train(xs[i], h)

    # TODO: make this not BPTT - which var to detach?
    for var in [xs[i], h[0], h[1]]:
        var = var.detach()

    h = h_1

    q_i = q_inference_output[i]
    loss = loss_function_z(zs[i], p_i, q_i)
    
    #TODO: how to fix?
    # loss.backward(retain_graph=True)
    # p_train_loss += loss.item()
    # p_optim.step()

# 3. use sampled z to run output cell in both networks
p_out_wrapper = OutputWrapper(input_size, hidden_size, length)
p_output = p_out_wrapper(xs, zs)

q_out_wrapper = OutputWrapper(input_size, hidden_size, length)
q_output = q_out_wrapper(xs, zs)

o = OutputCell(input_size, hidden_size)

# 4. Train loss over output
parameters = list(p_out_wrapper.parameters()) + list(q_out_wrapper.parameters())
out_train_loss = 0
optimizer = torch.optim.SGD(parameters, lr=lr)
for i in range(length):
    optimizer.zero_grad()
    p_i = p_output[i]
    q_i = q_output[i]

    # TODO: right variable to detach?
    zs[i].detach_()

    loss = loss_score_function(p_i, q_i)

    #TODO: shouldn't need the flag
    loss.backward(retain_graph=True)
    out_train_loss += loss.item()
    optimizer.step()

### Evaluation ###

p_output = torch.Tensor(p_output)

print(p_output[:,0,0])



















    


