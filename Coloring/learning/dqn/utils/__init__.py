import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image

import torch
from environment.wrappers import MultiagentWrapper
import torchvision.transforms as T

from learning.utils.env import make_env


# a named tuple representing a single transition in our environment
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# Pre-process
# extracting, processing rendered images from environment.
resize = T.Compose([T.ToPILImage(), T.Resize(
    40, interpolation=Image.CUBIC), T.ToTensor()])