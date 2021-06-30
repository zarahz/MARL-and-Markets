from abc import abstractmethod, abstractproperty
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
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        # generate 2d weights by choosing random numbers out of the
        # standard deviation given mean = 0 and std = 1
        m.weight.data.normal_(0, 1)
        # m.weight.data.shape = tensor(64,64)
        # für jedes gewicht x wird berechnet x/(sqrt(sum x^2))
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        # torch.sqrt ergibt shape (64,1), dh jede weight zeile wird durch ein torch.sqrt Value geteilt, siehe x oben
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            # Print(),
            # image passed to conv2d(in_channels, out_channels, kernel_size,...)
            # in_channels is 3 here for encoding (obj, stat, col) normally its rgb (or 1 for bw-images)
            # conv2d slides kernel with 1 values over image and sums up values -> creates a new matrix with summed values!
            nn.Conv2d(3, 16, (2, 2)),  # torch.Size([16, 16, 6, 6])
            # Print(),
            nn.ReLU(),  # output of conv2d is input of ReLU - ReLu neutralizes negative values
            # output of ReLU is used here and so on - MaxPooling takes highest values of 2x2 grid and reduces matrix size
            nn.MaxPool2d((2, 2)),  # torch.Size([16, 16, 3, 3])
            # Print(),
            nn.Conv2d(16, 32, (2, 2)),  # torch.Size([16, 32, 2, 2])
            # Print(),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),  # torch.Size([16, 64, 1, 1])
            # Print(),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(
                self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(
                obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(
                self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        # centralized critic decentralized actors = multiple actors with one critic?
        # so here the layer needs to be a conv1d?
        self.actor = nn.Sequential(
            # Print(),
            nn.Linear(self.embedding_size, 64),
            # Print(),
            nn.Tanh(),  # gruppiert werte zu -1 & 1 -> classification
            nn.Linear(64, action_space.n),
            # Print()
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        # obs.image shape is [3,16,7,7]
        # contains 16 (parallel envs) images with 3 channels and a size of agent-view (7x7 per default)
        x = obs.image.transpose(1, 3).transpose(2, 3)
        # here x.shape = [16,3,7,7] => [batch_size, channels, height, width]
        x = self.image_conv(x)
        # x has now for all 16 episodes the 64 out channels
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size],
                      memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # actor gets (64) values for each parallel env
        x = self.actor(embedding)
        # calculates the probalbilities of all actions in each parallel env? (shape [16,5])
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        # calculates all values in all envs? (shape [16,1] or [16] after squeeze)
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
