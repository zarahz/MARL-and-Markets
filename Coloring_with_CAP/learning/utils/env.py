import gym
# import gym_minigrid
from environment.wrappers import *


def make_env(env_key, agents, seed=None, percentage_reward=False, mixed_motive=False):
    env = gym.make(id=env_key, agents=agents)
    env = CooperativeMultiagentWrapper(env, percentage_reward, mixed_motive)
    env.seed(seed)
    return env
