import numpy as np
import numpy.random as r

'''
TODO:
    initialization,
    predict,
    backprop
'''

# # example architecture
# arch = {
#     'input_size':5,
#     'output_size':1,
#     'hidden_sizes':[4,4,4],
#     'activation':None,
#     'ddx_activation':None,
#     'concluding_activation':None,
#     'ddx_concluding_activation':None,
# }

class nn():
    def __setup__(self, architecture):
        '''parse architecture dictionary'''
        self.input_size = architecture['input_size']
        self.output_size = architecture['output_size']
        self.hidden_sizes = architecture['hidden_sizes']
        self.activation = architecture['activation']
        self.ddx_activation = architecture['ddx_activation']
        self.concluding_activation = architecture['concluding_activation']
        self.ddx_concluding_activation = architecture['ddx_concluding_activation']
    
    def __initparams__(self):
        # now we must randomly initialize the weights and biases with correct dimensionality.
        weights = {} # note: there are no params for the input layer
        bias = {}

        # init hidden layers one by one with random params

        weights['hidden1'] = r.randn(self.hidden_sizes[0], self.input_size)
        weights['hidden2'] = r.randn(self.hidden_sizes[1], self.hidden_sizes[0])
        weights['hidden3'] = r.randn(self.hidden_sizes[2], self.hidden_sizes[1])
        weights['out'] = r.randn(self.output_size, self.hidden_sizes[2])

        bias['hidden1'] = r.randn(self.hidden_sizes[0])
        bias['hidden2'] = r.randn(self.hidden_sizes[1])
        bias['hidden3'] = r.randn(self.hidden_sizes[2])
        bias['out'] = r.randn(self.output_size)

        # assign to instance variable
        self.params_ = {'weights':weights, 'bias':bias}
    
    def params(self):
        '''prints network params, starting with weights and then biases'''
        weights = self.params_['weights']
        bias = self.params_['bias']

        for key in weights.keys():
            print(f'{key}:\nw:{weights[key]},{weights[key].shape}\nb:{bias[key]},{bias[key].shape}\n\n')
    
    def predict(self, X, output='base'):
        '''performs a prediction given some input X, use output='verbose' to get entire cache '''
        cache = {}  # create cache to hold intermediary results
        weights = self.params_['weights'] # reduce my typing work
        bias = self.params_['bias'] # reduce my typing work x2


        cache['a0'] = X

        cache['a1'] = weights['hidden1'] @ cache['a0'] + bias['hidden1']
        cache['z1'] = self.activation(cache['a1'])

        cache['a2'] = weights['hidden2'] @ cache['z1'] + bias['hidden2']
        cache['z2'] = self.activation(cache['a2'])

        cache['a3'] = weights['hidden3'] @  cache['z2'] + bias['hidden3']
        cache['z3'] = self.activation(cache['a3'])

        cache['aout'] = weights['out'] @ cache['z3'] + bias['out']
        cache['zout'] = self.concluding_activation(cache['aout'])

        if (output=='base'):
            return cache['zout']
        elif (output == 'verbose'):
            return cache
        else:
            raise Exception('not a valid output type, use either \'base\' or \'verbose\' ')
    
    def backwards(self, X, Y_true, learning_rate=0.01):
        cache = self.predict(X, output='verbose')
        weights = self.params_['weights']
        bias = self.params_['bias']

        # Compute the loss
        loss = 0.5 * np.sum((cache['zout'] - Y_true)**2)

        # Compute gradients for the output layer
        dL_daout = cache['zout'] - Y_true
        daout_dzout = self.ddx_concluding_activation(cache['aout'])
        dzout_da3 = weights['out'].T
        dL_da3 = dL_daout * daout_dzout
        dL_dz3 = dL_da3 @ dzout_da3

        # Compute gradients for hidden layer 3
        da3_dz3 = self.ddx_activation(cache['a3'])
        dz3_da2 = weights['hidden3'].T
        dL_da2 = dL_dz3 * da3_dz3
        dL_dz2 = dL_da2 @ dz3_da2

        # Compute gradients for hidden layer 2
        da2_dz2 = self.ddx_activation(cache['a2'])
        dz2_da1 = weights['hidden2'].T
        dL_da1 = dL_dz2 * da2_dz2
        dL_dz1 = dL_da1 @ dz2_da1

        # Compute gradients for hidden layer 1
        da1_dz1 = self.ddx_activation(cache['a1'])
        dz1_da0 = weights['hidden1'].T
        dL_da0 = dL_dz1 * da1_dz1

        # Update parameters using gradients and learning rate
        weights['out'] -= learning_rate * (dL_da3 @ cache['z3'].T)
        weights['hidden3'] -= learning_rate * (dL_da2 @ cache['z2'].T)
        weights['hidden2'] -= learning_rate * (dL_da1 @ cache['z1'].T)
        weights['hidden1'] -= learning_rate * (dL_da0 @ cache['a0'].T)

        bias['out'] -= learning_rate * np.sum(dL_da3, axis=1)
        bias['hidden3'] -= learning_rate * np.sum(dL_da2, axis=1)
        bias['hidden2'] -= learning_rate * np.sum(dL_da1, axis=1)
        bias['hidden1'] -= learning_rate * np.sum(dL_da0, axis=1)

        # Return the computed loss
        return loss



    def __init__(self, architecture:dict) -> None:
        '''PLEASE NEVER CALL THE DUNDER FUNCTIONS EXPLICITY !!!'''
        self.__setup__(architecture) # should have all the important params stored now >:D
        self.__initparams__() # now we have initialized all weights/biases randomly :O
        