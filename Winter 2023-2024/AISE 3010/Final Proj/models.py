# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # fc layers
        self.fc1 = nn.Linear(310, 500) # input size is 310
        self.fc2 = nn.LazyLinear(500)
        self.fc3 = nn.LazyLinear(400)
        self.fc4 = nn.LazyLinear(150) 
        self.fc5 = nn.LazyLinear(3) # 3 classes (i think)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return x
    
