# imports
import argparse
import time
import gym
import torch
from environment.wrappers import CooperativeMultiagentWrapper

import learning.utils
from matplotlib import pyplot as plt

# preparation

# Set seed for all randomness sources
seed = 0
learning.utils.seed(seed)

one_agent_env = gym.make(id="Empty-Grid-v0", agents=1, max_steps=15)
one_agent_env = CooperativeMultiagentWrapper(one_agent_env)
one_agent_env.seed(seed + 10000)

two_agents_env = gym.make(id="Empty-Grid-v0", agents=2, max_steps=15)
two_agents_env = CooperativeMultiagentWrapper(two_agents_env)
two_agents_env.seed(seed + 10000)

three_agents_env = gym.make(id="Empty-Grid-v0", agents=2, max_steps=15)
three_agents_env = CooperativeMultiagentWrapper(three_agents_env)
three_agents_env.seed(seed + 10000)

print("Environments loaded\n")

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load agent

model_dir_one = learning.utils.get_model_dir("emptyGrid-one-agent")
model_dir_two = learning.utils.get_model_dir("emptyGrid-two-agents")
model_dir_three = learning.utils.get_model_dir("emptyGrid-three-agents")

one_agent = learning.utils.Agent(0, one_agent_env.observation_space,
                                 one_agent_env.action_space, model_dir_one, device=device)

two_agents = []
for double_agent in range(2):
    two_agents.append(learning.utils.Agent(double_agent,
                                           two_agents_env.observation_space, two_agents_env.action_space,
                                           model_dir_two, device=device))

three_agents = []
for triple_agent in range(2):
    three_agents.append(learning.utils.Agent(triple_agent,
                                             three_agents_env.observation_space, three_agents_env.action_space,
                                             model_dir_three, device=device))

print("Agents loaded\n")
