import numpy as np

class NN:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):

        # store locallt relevant information
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size

        # init weights randomly
        self.weights = {
            'W1': np.random.rand(input_size, hidden_size1),
            'W2': np.random.rand(hidden_size1, hidden_size2),
            'W3': np.random.rand(hidden_size2, hidden_size3),
            'W4': np.random.rand(hidden_size3, output_size)
        }

        # init biases randomly
        self.biases = {
            'b1': np.zeros((1, hidden_size1)),
            'b2': np.zeros((1, hidden_size2)),
            'b3': np.zeros((1, hidden_size3)),
            'b4': np.zeros((1, output_size))
        }

    # allow changing of activation function, must be called after class initialization
    def define_activation(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    # propagate input signals to output
    def predict(self, X):
        # create a 'cache' (dictionary lol) to track the layer inputs and outputs
        self.cache = {}

        # input -> layer 1
        self.cache['hidden1_input'] = np.dot(X, self.weights['W1']) + self.biases['b1']
        self.cache['hidden1_output'] = self.activation(self.cache['hidden1_input'])

        # layer 1 -> layer 2
        self.cache['hidden2_input'] = np.dot(self.cache['hidden1_output'], self.weights['W2']) + self.biases['b2']
        self.cache['hidden2_output'] = self.activation(self.cache['hidden2_input'])

        # layer 2 -> layer 3
        self.cache['hidden3_input'] = np.dot(self.cache['hidden2_output'], self.weights['W3']) + self.biases['b3']
        self.cache['hidden3_output'] = self.activation(self.cache['hidden3_input'])

        # layer 3 -> output
        self.cache['output_input'] = np.dot(self.cache['hidden3_output'], self.weights['W4']) + self.biases['b4']
        self.cache['output'] = self.activation(self.cache['output_input'])

        # final output
        return self.cache['output']

    def calculate_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backpropagation(self, X, y, learning_rate):
        m = X.shape[0]

        ## output layer

        # calculate error signal
        output_error = self.cache['output'] - y
        output_delta = output_error * self.activation_derivative(self.cache['output'])

        # output layer adjustments
        self.weights['W4'] -= learning_rate * np.dot(self.cache['hidden3_output'].T, output_delta)
        self.biases['b4'] -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        ## didden layer 3
        hidden3_error = np.dot(output_delta, self.weights['W4'].T)
        hidden3_delta = hidden3_error * self.activation_derivative(self.cache['hidden3_output'])
        self.weights['W3'] -= learning_rate * np.dot(self.cache['hidden2_output'].T, hidden3_delta)
        self.biases['b3'] -= learning_rate * np.sum(hidden3_delta, axis=0, keepdims=True)

        # Hidden layer 2
        hidden2_error = np.dot(hidden3_delta, self.weights['W3'].T)
        hidden2_delta = hidden2_error * self.activation_derivative(self.cache['hidden2_output'])
        self.weights['W2'] -= learning_rate * np.dot(self.cache['hidden1_output'].T, hidden2_delta)
        self.biases['b2'] -= learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)

        # Hidden layer 1
        hidden1_error = np.dot(hidden2_delta, self.weights['W2'].T)
        hidden1_delta = hidden1_error * self.activation_derivative(self.cache['hidden1_output'])
        self.weights['W1'] -= learning_rate * np.dot(X.T, hidden1_delta)
        self.biases['b1'] -= learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation (predictions)
            predictions = self.predict(X)

            # Calculate loss
            loss = self.calculate_loss(y, predictions)

            # Backward propagation
            self.backpropagation(X, y, learning_rate)

            if epoch % 10000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')


