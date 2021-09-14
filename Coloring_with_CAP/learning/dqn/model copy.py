import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, observation_size, action_space):
        super(DQN, self).__init__()
        # observation_size -> all shape values multiplied
        # outputs -> number of possible actions
        pre_head_dim = 32  # 16
        self.fc_net = nn.Sequential(
            nn.Linear(observation_size["image"][0], 32),
            nn.ELU(),
            nn.Linear(32, pre_head_dim),
            nn.ELU()
        )
        self.action_head = nn.Linear(pre_head_dim, action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        # x = x.view(x.size(0), -1)
        x = self.fc_net(x)
        return self.action_head(x)

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        # self.bn1 = nn.Identity(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        # self.bn2 = nn.Identity(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2)
        # self.bn3 = nn.Identity(32)

        # w = observation_size["image"][0]
        # h = observation_size["image"][1]
        # # Number of Linear input connections depends on output of conv2d layers
        # # and therefore the input image size, so compute it.

        # def conv2d_size_out(size, kernel_size=1, stride=2):
        #     return (size - (kernel_size - 1) - 1) // stride + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     return self.head(x.view(x.size(0), -1))
