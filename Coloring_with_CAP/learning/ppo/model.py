from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass


class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs):
        pass

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, RecurrentACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

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

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
