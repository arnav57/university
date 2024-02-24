# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x.shape = 4, 16, 5, 5
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Conv related layers
        self.pool = nn.MaxPool2d(3,1)
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,12,5)
        self.conv3 = nn.Conv2d(12,24,5)
        self.conv4 = nn.Conv2d(24, 36, 5)
        self.conv5 = nn.Conv2d(36, 50, 5)
        # Dense layers
        self.fc1 = nn.Linear(24 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 30)
        self.fc4 = nn.Linear(30,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x))) # conv2 -> relu -> pool
        x = self.pool(F.relu(self.conv3(x))) # conv3 -> relu -> pool

        x = torch.flatten(x,1) # flatten before dense layers

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

class Net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Conv related layers
        self.pool = nn.MaxPool2d(3,1)
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,12,3)
        self.conv3 = nn.Conv2d(12,24,3)
        self.conv4 = nn.Conv2d(24, 36, 3)
        self.conv5 = nn.Conv2d(36, 50, 3)
        # Dense layers
        self.fc1 = nn.Linear(50 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 30)
        self.fc4 = nn.Linear(30,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x))) # conv2 -> relu -> pool
        x = self.pool(F.relu(self.conv3(x))) # conv3 -> relu -> pool
        x_midway = x # store current x in x_midway -> 32, 24, 20, 20
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # print(x_midway.shape)
        x = torch.flatten(x,1) # flatten before dense layers

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x, torch.flatten(x_midway, 1) # return actual output and midway output but flattened

class MidwayClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(24 * 20 * 20, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return x