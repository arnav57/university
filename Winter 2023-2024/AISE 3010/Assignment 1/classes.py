import numpy as np

class neuralnet():

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu(self, z):
        return np.maximum(0,z)
    
    def relu_derivative(self, z):
        return z > 0
    
    def softmax(self, z):
        A = np.exp(z) / sum(np.exp(z))
        return A

    def init_params(self):
        # input will be of size num_feats x 1
        # h1
        self.w1 = np.random.randn(10, self.num_feats) # (out_size=10 x 784)(784 x 1) + (out_size x 1) = (10, 1)
        self.b1 = np.random.randn(10, 1)
        # h2
        self.w2 = np.random.randn(10, 10) # (10, 10)x(10 ,1) + (10,1)
        self.b2 = np.random.randn(10, 1)
        # h3
        self.w3 = np.random.randn(10, 10) # (10, 10)x(10 ,1) + (10,1)
        self.b3 = np.random.randn(10, 1)
        # output
        self.w4 = np.random.randn(self.out_size, 10) # (out, 10)(10 ,1) + (out,1)
        self.b4 = np.random.randn(self.out_size, 1)

    def __init__(self, num_feats, out_size, type):
        self.num_feats = num_feats
        self.out_size = out_size
        self.type = type
        self.init_params()

    def forward(self, x):
        fc = {}
        fc['in'] = x

        fc['z1'] = np.dot(self.w1, x) + self.b1
        fc['a1'] = self.sigmoid(fc['z1'])

        fc['z2'] = np.dot(self.w2, fc['a1']) + self.b2
        fc['a2'] = self.sigmoid(fc['z2'])

        fc['z3'] = np.dot(self.w3, fc['a2']) + self.b3
        fc['a3'] = self.sigmoid(fc['z3'])

        fc['z4'] = np.dot(self.w4, fc['a3']) + self.b4

        if (self.type == 'classifier'):
            fc['a4'] = self.softmax(fc['z4']) # concluding softmax activation for classification
        else:
            fc['a4'] = fc['z4']

        self.fc = fc

        return fc['a4']
    
    def update_params(self, dw1, db1, dw2, db2, dw3, db3, dw4, db4, alpha):
        self.w1 -= alpha * dw1
        self.b1 -= alpha * db1
        self.w2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.w3 -= alpha * dw3
        self.b3 -= alpha * db3
        self.w4 -= alpha * dw4
        self.b4 -= alpha * db4
    
    def backward(self, x, y, lr):
        '''y must be one hot encoded'''
        _ = self.forward(x) # update fc
        fc = self.fc # obtain fc

        dz4 = fc['a4'] - y
        dw4 = np.dot(dz4, fc['a3'].T)
        db4 = np.sum(dz4)

        dz3 = np.dot(self.w4.T, dz4) * self.sigmoid_derivative(fc['z3'])
        dw3 = np.dot(dz3, fc['a2'].T)
        db3 = np.sum(dz3)

        dz2 = np.dot(self.w3.T, dz3) *  self.sigmoid_derivative(fc['z2'])
        dw2 = np.dot(dz2, fc['a1'].T)
        db2 = np.sum(dz2)

        dz1 = np.dot(self.w2.T, dz2) * self.sigmoid_derivative(fc['z1'])
        dw1 = np.dot(dz1, fc['in'].T)
        db1 = np.sum(dz1)

        self.update_params(dw1, db1, dw2, db2, dw3, db3, dw4, db4, alpha=lr)

def one_hot_encode(Y):
    one_hot_Y = np.zeros((10,1))
    one_hot_Y[Y] = 1
    return one_hot_Y