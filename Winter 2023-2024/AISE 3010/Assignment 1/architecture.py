import numpy as np 

# classes that build an NN
## Neurons -> Layers -> Networks

class neuron():
    # should be defined with num_inputs and activation function, should randomly create weights/bias upon instantiation
    def __init__(self, num_inputs, activation):
        self.num_inputs = num_inputs
        self.activation = activation
        self.__setup__()

    # organization of constructor
    def __setup__(self):
        # create a list of size to hold inputs ; populate weights & bias randomly upon initialization
        self.inputs = [None for i in range(self.num_inputs)] # inputs NEED to be set using 'set_inputs()'
        self.weights = [np.random.rand(1) for i in range(self.num_inputs)]
        self.bias = np.random.rand(1)
        self.__setparams__()

    # update params_ field
    def __setparams__(self):
        self.params_ = [self.weights, self.bias]

    # calculate output
    def __calcluate__(self):
        self.output = self.activation(np.sum(self.weights * self.inputs) + self.bias)

    def set_weights(self, new_weights):
        self.weights = new_weights
        self.__setparams__()
    
    def set_bias(self, new_bias):
        self.bias = new_bias
        self.__setparams__()
    
    def set_inputs(self, new_inputs):
        self.inputs = new_inputs
        # when new inputs are provided, calculate the output
        self.__calcluate__()

class inputlayer():
    # defined with a number of neurons and layer activation function
    def __init__(self, num_neurons, activation):
        self.num_neurons = num_neurons
        self.activation = activation
        self.__setup__()
    
    # constrcutor organization
    def __setup__(self):
        # create all the required neurons, each neuron accepts only 1 input as this is the input layer.
        self.neurons = [neuron(1, self.activation) for i in range(self.num_neurons)]
        self.__setparams__()

    def __calcluate__(self):
        # obtain layer output as a list
        self.output = [n.output for n in self.neurons]

    def __setparams__(self):
        self.params_ = []
        for n in self.neurons:
            self.params_.append(n.params_)
    
    def __summary__(self):
        print(f'Layer Input:{self.input}\nLayer Params:{self.params_}\nLayer Output:{self.output}')


    def set_input(self, new_input):
        # provide each neuron with an input
        self.input = new_input
        for i in range(self.num_neurons):
            self.neurons[i].set_inputs(np.array(self.input[i]))
        
        # upon providing a new layer input, calculate layer output
        self.__calcluate__()
        



        
        

    

        
        
