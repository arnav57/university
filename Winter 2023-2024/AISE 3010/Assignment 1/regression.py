from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from classes import *

nn = neuralnet(num_feats=8, out_size=1, type='regressor')

x,y = fetch_california_housing(return_X_y=True)

s = StandardScaler()
s.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

num_samples = x_train.shape[0]
epochs = 250
batch_size = 1000

iterations = int(num_samples/batch_size) + 1

print(x_train.shape)

epoch_losses = []
its = []

for epoch in range(epochs):
    # create a minibatch with # of samples = iterations
    indices = np.random.choice(range(x_train.shape[0]), batch_size, replace=False)
    x_batch = x_train[indices]
    y_batch = y_train[indices]

    batch_losses = []
    
    for i in range(iterations):
        x_d = x_batch[i].reshape(8,1)
        y_d = [y_batch[i]]

        preds = nn.forward(x_d)

        loss = mse(y_d, preds)
        batch_losses.append(loss)

        nn.backward(x_d, y_d, lr=1)
    
    avg_bloss = np.mean(batch_losses)
    epoch_losses.append(avg_bloss)
    its.append(epoch)
    print(f'Epoch: {i} - Batch MSE: {avg_bloss}')

plt.plot(its, epoch_losses)
plt.xlabel('iteration')
plt.ylabel('batch mse loss')
plt.title('training curve')
plt.show()