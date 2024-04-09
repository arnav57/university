# ROOT.py
# holds the utility, train and test functions for the general training script
# 

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm # progress bar
from models import * # import cnn architectures from other module
import sys, os, json # used to guarantee filepath and hyperparams stuff work

from dataset import EEGTestingData, EEGTrainingData, eegpca
from torch.utils.data import DataLoader

# define device as cuda if we got it
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# read hyperparameters from 'hyperparams.json', add or tweak them by altering+saving the file
j = open('hyperparams.json')
hparams = json.load(j)

# data preproc here
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print(f"Loading EEG Dataset (ignore the errors) ...\n\n\n")

trainset = EEGTrainingData(transform=transform)
testset = EEGTestingData(transform=transform)
eegpca(trainset=trainset, testset=testset)

trainloader = DataLoader(trainset, batch_size=hparams['batch_size'], shuffle=True)
testloader = DataLoader(testset, batch_size=hparams['batch_size'], shuffle=False)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=hparams['batch_size'],
#                                           shuffle=True)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=hparams['batch_size'],
#                                          shuffle=False)



# model, loss fn, optimizer
net = NeuralNet() # define new arhictectures in models.py
net = net.to(device)
criterion = nn.CrossEntropyLoss() # find a loss function
optimizer = optim.SGD(net.parameters(), lr=hparams['learning_rate'], momentum=hparams['momentum'])

# training loop
def train():
    epoch_losses = []
    iter_losses = []
    printInfo()
    print(f"{'BEGIN TRAINING LOOP':=^100}")
    for epoch in range(hparams['num_epochs']):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{hparams['num_epochs']} ; Batch Progress", unit=' batches', unit_divisor=hparams['batch_size']), 0):
            # train chosen network architecture
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs= net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iter_losses.append(loss.item())

        epoch_losses.append(running_loss)
    print(f"{'END TRAINING LOOP':=^100}")
    print(f"Training has finished, Please see the training curve...")
    plot(epoch_losses)

    
def printInfo():
    print(f"\n\nTraining a '{net.__class__.__name__}' model, on '{device}' device, with the following hyperparamters:")
    print(f"\t - {hparams['num_epochs']} loops over the dataset (epochs)")
    print(f"\t - batch size of {hparams['batch_size']}")
    print(f"\t - with '{optimizer.__class__.__name__}' optimizer:")
    print(f"\t\t - learning rate of {hparams['learning_rate']}")
    print(f"\t\t - momentum of {hparams['momentum']}")

def plot(epoch_losses):
    statistic = epoch_losses # plot epoch losses

    plt.plot(statistic, 'orange') # actual line
    plt.plot(statistic, 'bo') #dots
    plt.title('Total Epoch Cross Entropy Loss vs. Epoch Number')
    plt.xlabel('Minibatch Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

def save_weights():
    # check if directory '../checkpoints/' exists, if not create one
    exists = os.path.isdir('checkpoints')
    if not exists:
        os.mkdir('checkpoints')


    choice = input('\n\nWould you like to create or overwrite the current train checkpoint with the new weights?\n1 - Save weights\n2 - Proceed without Saving weights\nPlease Enter An Option: ')
    if (choice == '1'):
        current_path = os.getcwd()
        path = current_path + f'/checkpoints/{net.__class__.__name__}.pth'
        print(f"Saving weights to '{path}'")
        torch.save(net.state_dict(), path)
    else:
        print("Proceeding without saving weights...")

def load_weights():
    # check if directory exists
    path = f'checkpoints/{net.__class__.__name__}.pth'
    exists = os.path.isfile(path)
    if exists:
        choice = input(f"\n\nWould you like to load a train checkpoint for this architecture?\n1 - Start from checkpoint at '{path}'\n2 - Start without checkpoint\nPlease Enter An Option: ")
        if (choice == '1'):
            net.load_state_dict(torch.load(path))
            print(f"\nLoaded weights from checkpoint '{path}'...")


def test():
    # markers to keep track of statistics
    correct = 0
    total = 0

    print(f"Testing model with architecture: '{net.__class__.__name__}'\n") # prints the class name
    print(f"{'BEGIN TESTING LOOP':=^100}")

    net.eval()

    # dont need grad for testing
    with torch.no_grad():
        for data in tqdm(testloader, desc=f'Testing Progress', unit=' batches'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs= net(images)
            _ , predicted = torch.max(outputs.data, 1) # find max class index along dim 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"\n{'END TESTING LOOP':=^100}")
    print(f'\nTesting Accuracy: {100 * (correct / total):.2f} %')


def main():
    print(f"\n\nUsing architecture '{net.__class__.__name__}'... ")
    load_weights()
    choice = input("\n\nWould you like to train or test?\n1 - Train Only\n2 - Test Only\nPlease Enter An Option: ")
    if (choice == '1'):
        # train only
        train()
        test()
        save_weights()
    else:
        test()

if __name__ == "__main__":
    main()