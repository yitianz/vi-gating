import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

import p_network
import q_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data ###
# x: {0,1}
# y:{0,1} 
# y = xor(x) -> 0 until the 1st x, then flip to 1
LENGTH = 10

x = np.random.randint(0, 2, size=LENGTH)
zeros = np.zeros(LENGTH)
index = np.where(x == 1)[0][0]
y = [0 for _ in range(index)] + [1 for _ in range(len(x) - index)]
x = torch.Tensor(x).unsqueeze(1).unsqueeze(2).to(device)
y = torch.Tensor(y).unsqueeze(1).unsqueeze(2).to(device)
# print('x: {}'.format(x.size()))
# print('y: {}'.format(y))


### Loss functions ###

def loss_function_z(z, p_output, q_output):
    p_term = z*np.log(p_output) + (1-z)*np.log(p_output)
    q_term = z*np.log(q_output) + (1-z)*np.log(q_output)
    return p_term - q_term

def loss_function_y(y_pred, y_true):
    return nn.MSELoss()(y_pred, y_true)

### Training ###
INPUT_SIZE = 1
HIDDEN_SIZE = 2
lr = 1e-3
p_model = p_network.PNetwork(INPUT_SIZE, HIDDEN_SIZE).to(device)
q_model = q_network.QNetwork(INPUT_SIZE, HIDDEN_SIZE).to(device)
parameters = list(p_model.parameters()) + list(q_model.parameters())
optimizer = torch.optim.SGD(parameters, lr=lr)
# for each datapoint (x, y):
#    sample z | x, y ~ q_\phi
#    take gradient step for \theta and \phi of loss log (p_\theta(y, z | x) / q_\phi(z | x, y) )
p_model.train()
q_model.train()

q_output, _ = q_model(x, y)
p_output, _ = p_model(x)

train_loss = 0

for i in range(LENGTH):
    optimizer.zero_grad()
    z = q_model.sample(0, i)
    # loss_z = loss_z(z, p_output, q_output)
    p_i = p_output[i]
    q_i = q_output[i]
    loss = torch.log(p_i / q_i) * torch.log(q_i)
    loss.backward()

    train_loss += loss.item()
    optimizer.step()





    


