import numpy as np
from architecture import NN

# activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# activation derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))               

# params
input_size = 2
hidden_size1 = 4
hidden_size2 = 3
hidden_size3 = 2
output_size = 1
epochs = 100001
learning_rate = 0.2

# generate sample data
np.random.seed(0)
X = np.random.rand(100, input_size)
y = np.random.randint(0, 2, (100, output_size))

# create and train the neural network
neural_net = NN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
neural_net.define_activation(sigmoid, sigmoid_derivative)
neural_net.train(X, y, epochs, learning_rate)