# the pain of 18 imports
from cnn import *
import numpy as np
import torch
import torch.functional as f
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm # i need a progress bar this shit takes ages man
import matplotlib.pyplot as plt

## choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## make cnn from cnn.py, we can swap model out for a diff architecture here
cnn = CNN1(10).to(device)

## get train/test datasets
# create transform to apply to images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# actually get the data lol
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)

# data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# define loss and (sgdm) opt
loss = nn.CrossEntropyLoss()
opt = optim.SGD(cnn.parameters(), lr=5e-1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.8) # every 1 epoch the lr <= 0.8*lr

# training loop
total_loss = []
center_msg('======================~TRAINING~======================')
num_epochs = 2
for epoch in range(num_epochs):
    epoch_loss = []
    # iterate over batches
    for inputs, labels in tqdm(train_loader, desc=f'Train Epoch: {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device) # move input and label batches to cuda
        opt.zero_grad() # reset gradients
        preds = cnn(inputs) # obtain preds
        loss_val = loss(preds, labels) # calculate CE loss
        loss_val.backward() # calculate gradients
        opt.step() # update parameters
        epoch_loss.append(loss_val.item())
        total_loss.append(loss_val.item())
    print(f'Epoch {epoch+1} // Initial Loss: {epoch_loss[0]} // Final Loss: {epoch_loss[-1]} // Avg Loss: {np.mean(epoch_loss)}') # print initial loss, end loss, and avg loss per epoch

# plot training curve 
plt.plot(total_loss)
plt.show()
# test loop
center_msg('=======================~TESTING~=======================')
correct = 0
total = 0
for inputs, labels in tqdm(test_loader, desc='Testing', unit='batch'):
    inputs, labels = inputs.to(device), labels.to(device) # move batch to cuda
    preds = cnn(inputs) # obtain preds
    _ , pred_class = torch.max(preds.data, 1) # obtain predicted class value

    # update total/correct
    total += labels.size(0)
    correct += (pred_class == labels).sum().item()

acc = correct/total
print(f'Testing Accuracy: {100*acc:.2f}%')
