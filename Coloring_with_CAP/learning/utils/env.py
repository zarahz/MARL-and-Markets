import gym
# import gym_minigrid
from environment.wrappers import *


def make_env(env_key, agents, grid_size=5, percentage_reward=False, mixed_motive=False, max_steps=None, seed=None):
    env = gym.make(id=env_key, agents=agents,
                   size=grid_size, max_steps=max_steps)
    env = MultiagentWrapper(env, percentage_reward, mixed_motive)
    env.seed(seed)
    return env
