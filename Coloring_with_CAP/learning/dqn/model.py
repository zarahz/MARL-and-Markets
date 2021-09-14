import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.embedding_size = size
        kernel_size = 2
        if size == 0:
            self.embedding_size = 64
            kernel_size = 1

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (kernel_size, kernel_size)),
            nn.ReLU()
        )
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )

    def forward(self, obs):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)

        return x
