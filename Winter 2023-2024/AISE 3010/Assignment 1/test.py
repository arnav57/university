import math
import numpy as np
import numpy.random as r
from architecture import nn



# example architecture, and activations

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
  return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # To avoid numerical instability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def softmax_p(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s,s)
  

arch = {
    'input_size':3,
    'output_size':2,
    'hidden_sizes':[2,3,2],
    'activation':sigmoid,
    'ddx_activation':sigmoid_p,
    'concluding_activation':softmax,
    'ddx_concluding_activation':softmax_p,
}

network = nn(architecture=arch)
network.params()
print(network.predict(r.randn(3), output='base'))
