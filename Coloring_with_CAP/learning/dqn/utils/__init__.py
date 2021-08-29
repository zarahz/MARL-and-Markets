import gym
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image

import torch
from environment.wrappers import MultiagentWrapper
import torchvision.transforms as T

# env = gym.make('CartPole-v0').unwrapped
env = gym.make(id='Empty-Grid-v0', agents=1, size=5,
               market="", trading_fee=0.05, max_steps=None)
# whereas settings of percentage reward or mixed motive only influences the reward outcome and therefore only needed in the wrapper
env = MultiagentWrapper(env)
env.seed(1)
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a named tuple representing a single transition in our environment
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# Pre-process
# extracting, processing rendered images from environment.
resize = T.Compose([T.ToPILImage(), T.Resize(
    40, interpolation=Image.CUBIC), T.ToTensor()])
screen_width = 600
view_width = 320
