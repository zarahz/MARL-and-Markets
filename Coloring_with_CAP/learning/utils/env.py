import gym
# import gym_minigrid
from environment.wrappers import *


def make_env(env_key, agents, seed=None):
    env = gym.make(id=env_key, agents=agents)
    env = CooperativeMultiagentWrapper(env)
    env.seed(seed)
    return env
