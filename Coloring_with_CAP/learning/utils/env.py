import gym
# import gym_minigrid
from environment.wrappers import *


def make_env(env_key, agents, grid_size=5, percentage_reward=False, mixed_motive=False, market="", trading_fee=0.05,  max_steps=15, seed=None):
    # markets influence action space and therefore needed in the environment directly
    env = gym.make(id=env_key, agents=agents, size=grid_size,
                   market=market, trading_fee=trading_fee, max_steps=max_steps)
    # whereas settings of percentage reward or mixed motive only influences the reward outcome and therefore only needed in the wrapper
    env = MultiagentWrapper(env, percentage_reward, mixed_motive)
    env.seed(seed)
    return env
