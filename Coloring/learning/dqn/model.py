import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.fc4 = nn.Linear(64*4*4, 1024)
        self.head = nn.Linear(1024, action_space)

    def forward(self, obs):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)
