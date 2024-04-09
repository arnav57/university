# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # fc layers
        self.fc1 = nn.Linear(66, 128) # input size is 310
        self.norm1 = nn.LazyBatchNorm1d()
        self.fc2 = nn.LazyLinear(256)
        self.norm2 = nn.LazyBatchNorm1d()
        self.fc3 = nn.LazyLinear(512)
        self.norm3 = nn.LazyBatchNorm1d()
        self.fc4 = nn.LazyLinear(1024)
        self.norm4 = nn.LazyBatchNorm1d()
        self.fc5 = nn.LazyLinear(128)
        self.norm5 = nn.LazyBatchNorm1d()
        self.fc6 = nn.LazyLinear(3) # 3 classes (i think)


    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        x = F.relu(self.norm4(self.fc4(x)))
        x = F.relu(self.norm5(self.fc5(x)))
        x = F.softmax(self.fc6(x), dim=1) # take softmax along data dim. dim0 is batch dim

        return x