# module imports
from models import *
from aise3010_assignment2_q1 import BATCH_SIZE, EPOCHS, LR, MTM # import hyperparams from q1
from aise3010_assignment2_q1 import trainloader, testloader, criterion # import dataloaders from q1
from aise3010_assignment2_q1 import plot # import plot fcn

# pkg imports
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
import sys, os # used to guarantee filepath stuffs work

# define globals
device = ('cuda' if torch.cuda.is_available() else 'cpu')
net = Net2()
net = net.to(device)

clf = MidwayClassifier()
clf = clf.to(device)

optimizer = torch.optim.SGD(clf.parameters(),  lr=LR, momentum=MTM)

try:
    # obtain current training checkpoint
    current = os.getcwd()
    PATH = current + f"/checkpoints/{net.__class__.__name__}.pth"
    # load CNN from Net2 training checkpoint
    net.load_state_dict(torch.load(PATH))
    print(f"Loaded Training Checkpoint for model: '{net.__class__.__name__}' from '{PATH}'\n\n")
except:
    print(f"Go into script for q1 and save a training checkpoint for model '{net.__class__.__name__}'\n\n")


def train_classifier():
    epoch_losses = []
    print(f"{'BEGIN TRAINING LOOP':=^100}") # center msg
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}/{EPOCHS} ; Batch Progress', unit=' batches', unit_divisor=BATCH_SIZE), 0):
            # move to gpu
            inputs, labels =  data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # get midway features
            outputs , midways = net(inputs)
            midways = midways.detach() # detach grad from midway, we only care about the grad from the midwayclassifier not original CNN
            # propagate midways through clf network
            preds = clf(midways)
            # obtain loss
            loss = criterion(preds, labels)
            running_loss += loss.item()
            loss.backward()
            # iterate params
            optimizer.step()

        epoch_losses.append(running_loss)
    print(f"{'END TRAINING LOOP':=^100}") # center msg    
    print('\n\nFinished Training ... See the training curve (close it to continue).\n')
    plot(epoch_losses)

    q = input("Would you like to save the weights as a checkpoint?\n1 - Save weights\n2 - Discard weights\n\nPlease Enter An Option: ")
    if q == '1':
        current = os.getcwd()
        PATH = current + f'/checkpoints/{clf.__class__.__name__}.pth'
        print(f"Saving Weights to: '{PATH}'")
        torch.save(clf.state_dict(), PATH)
    else:
        sys.exit(0) # exit without error.

def test_classifier():
    # hold metrics
    correct = 0
    total = 0


    # models to eval mode
    net.eval()
    clf.eval()
    print(f'\n{"BEGIN TESTING LOOP":=^100}')
    for i, data in enumerate(tqdm(testloader, desc=f'Test Batch ; Batch Progress', unit=' batches', unit_divisor=BATCH_SIZE), 0):
        with torch.no_grad():
            # move to gpu
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # get midway features
            outputs, midways = net(inputs)
            preds = clf(midways) # these are the actual outputs from clf
            _ , predicted = torch.max(preds.data, 1) # find max class index along dim 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'\n{"END TESTING LOOP":=^100}')
    print(f'\nTesting Accuracy: {100 * (correct / total):.2f} %')

            

def main():
    choice = input('Would you like to load the standalone classifier from a checkpoint or train from scratch?\n1 - Start from a checkpoint\n2 - Start from scratch\n\nPlease Enter An Option: ')
    if choice == '1':
        # load from checkpoint
        current = os.getcwd()
        PATH = current + f'/checkpoints/{clf.__class__.__name__}.pth'
        clf.load_state_dict(torch.load(PATH))
        print(f"\nLoaded checkpoint from '{PATH}'\n")
        cho = input("Would you like to test or train the standalone classifier from this checkpoint?\n1 - Test from this checkpoint\n2 - Train from this checkpoint\n\nPlease Enter An Option: ")
        if cho == '1':
            test_classifier()
        elif cho == '2':
            train_classifier()
            test_classifier()
        else:
            sys.exit(1) # else exit with error
    elif choice == '2':
        train_classifier()
        test_classifier()
        

if __name__ == '__main__':
    main()