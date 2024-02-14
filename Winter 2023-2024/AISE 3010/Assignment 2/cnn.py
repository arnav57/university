import torch.nn as nn
import os

class CNN1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(15, 30, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(30, 60, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten() # after conv layers we need to flatten it before feeding it into fully-connected layers

        self.fc = nn.Sequential(
            nn.Linear(60 * 4 * 4, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def center_msg(string):
    cmd_width = os.get_terminal_size().columns
    padding = (cmd_width - len(string)) // 2
    centered_string = " " * padding + string
    print(centered_string, '\n')