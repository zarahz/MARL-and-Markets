import gym
# import gym_minigrid
from environment.wrappers import *


def make_env(env_key, agents, grid_size=5, agent_view_size=7, setting="", market="", trading_fee=0.05,  max_steps=None, seed=None):
    # markets influence action space and therefore needed in the environment directly
    competitive = "competitive" in setting
    env = gym.make(id=env_key, agents=agents, size=grid_size, competitive=competitive, agent_view_size=agent_view_size,
                   market=market, max_steps=max_steps)
    # whereas settings of percentage reward or mixed motive only influences the reward outcome and therefore only needed in the wrapper
    env = MultiagentWrapper(env, setting, trading_fee)
    env.seed(seed)
    return env
