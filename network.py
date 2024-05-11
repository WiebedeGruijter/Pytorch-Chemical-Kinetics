
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 10, kernel_size=(11))
        self.conv2 = nn.Conv1d(10, 20, kernel_size=(12), padding=5)
        self.flatten = nn.Flatten(start_dim=0)
        self.linear3 = nn.Linear(13660, 13350)
        
    def forward(self, x):
        # input 1x2744, output 10x2734
        x = self.conv1(x)
        # input 10x2734, output 10x1367
        x = F.max_pool1d(x, kernel_size=(2))
        x = F.relu(x)
        # input 10x1367, output 20x1368
        x = self.conv2(x)
        # input 20x1368, output 20x683
        x = F.max_pool1d(x, kernel_size=(2))
        x = F.relu(x)
        # input 20x683, output 13660
        x = self.flatten(x)
        # input 13660, output 13350
        x = self.linear3(x)
        x = F.relu(x)
        # input 13350, output 1x150x89
        x = torch.reshape(x, (1, 150, 89))

        return x