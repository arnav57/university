# imports
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

# define device for training
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# define hyperparams

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MTM = 9e-1

# create datasets and dataloaders

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = BATCH_SIZE

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)




# model, loss func and optimizer
net = Net2()
net = net.to(device)
classifier = MidwayClassifier()
classifier = classifier.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MTM)
moptimizier = optim.SGD(classifier.parameters(), lr=LR, momentum=MTM)

# training loop
def train():
    epoch_losses = []
    iter_losses = []
    print(f"Training model w/ arhictecture '{net.__class__.__name__}' over {EPOCHS} loops of the dataset, with batch size of {BATCH_SIZE}")
    print(f'Using {optimizer.__class__.__name__} optimizer w/ learning rate: {LR}, momentum: {MTM}')
    print(f'Note: These hyperparameters can be easily changed in the script\n')
    trainmsg = 'BEGIN TRAINING LOOP'
    endtrainmsg = 'END TRAINING LOOP'
    print(f'{trainmsg:=^100}') # this is a fancy way to center  a message within 100 spaces
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}/{EPOCHS} ; Batch Progress', unit=' batches', unit_divisor=BATCH_SIZE), 0):
            # train chosen network architecture
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, midway = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iter_losses.append(loss.item())

        epoch_losses.append(running_loss)

    print(f'\n{endtrainmsg:=^100}')
    print('\nFinished Training, See the Training Curve Plot...')

    plot(epoch_losses)
    print()
    q = input('Would you like to create or overwrite current saved train checkpoint before proceeding?\n1 - Save weights\n2 - Do not save weights\n\nPlease Enter An Option: ')
    if q == '1':
        current = os.getcwd()
        PATH = current + f'/checkpoints/{net.__class__.__name__}.pth'
        print(f"Saving Weights to: '{PATH}'")
        torch.save(net.state_dict(), PATH)
    else:
        print("Proceeding without Saving weights...")

# plotting training curve(s), to see the curve uncomment 'plt.show()'
def plot(epoch_losses):
    statistic = epoch_losses # plot epoch losses

    plt.plot(statistic, 'orange') # actual line
    plt.plot(statistic, 'bo') #dots
    plt.title('Total Epoch Cross Entropy Loss vs. Epoch Number')
    plt.xlabel('Minibatch Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

# testing loop
def test():
    # markers to keep track of statistics
    correct = 0
    total = 0

    print(f"Testing model with architecture: '{net.__class__.__name__}'\n") # prints the class name
    testmsg = 'BEGIN TESTING LOOP'
    endtestmsg = 'END TESTING LOOP'
    print(f'{testmsg:=^100}')

    net.eval()
    midways = []
    mlabels = []
    # dont need grad for testing
    with torch.no_grad():
        for data in tqdm(testloader, desc=f'Testing Progress', unit=' batches'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, midway = net(images)
            midway.to(device)
            midways.append(midway)
            mlabels.append(labels)
            _ , predicted = torch.max(outputs.data, 1) # find max class index along dim 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'\n{endtestmsg:=^100}')
    print(f'\nTesting Accuracy: {100 * (correct / total):.2f} %')


    

def main():
    choice = input(f"\n\nWould you like to use a train checkpoint for '{net.__class__.__name__}'?\n1 - Start from checkpoint\n2 - to train from scratch\n\nPlease Enter An Option: ")
    print(f"Note: Proceeding with the '{device}' device\n\n\n")
    if choice == '1':
        current = os.getcwd()
        PATH = current + f'/checkpoints/{net.__class__.__name__}.pth'
        print(f"Loading Weights from: '{PATH}'")
        # load train checkpoint
        net.load_state_dict(torch.load(PATH))
        print("Loaded Train Checkpoint ...")
        cho = input('\n\nWould you like to further train from this checkpoint, or test from this checkpoint?\n1 - Test from this checkpoint\n2 - Train from this checkpoint\n\nPlease Enter An Option: ')
        if cho == '1':
            test()
        elif cho == '2':
            train()
            test()
        else:
            sys.exit(1) # exit with error
    elif choice =='2':
        train()
        test()

if __name__ == '__main__':
    main()