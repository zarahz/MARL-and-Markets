import gym
from gym import spaces
from environment.colors import *

import environment


class CooperativeMultiagentWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        if done:  # adjust reward to be calculated out of percentage
            reward = self.calculate_reward()
        return observation, reward, done, info

    def calculate_reward(self):
        # reward based on completed coloring
        if self.env.whole_grid_colored():
            print('---- GRID FULLY COLORED! ----')
            return 1
        return 0

        # reward based on coloring percentage
        # return 1 * self.env.grid_colored_percentage()

    def print_coloring_data(self):
        floor_tiles = 0
        colored_tiles = 0
        colored_by_agent = {}
        for obj in self.env.grid.grid:
            if isinstance(obj, environment.grid.Floor):
                floor_tiles += 1
                if obj.is_colored:
                    colored_tiles += 1
                    if obj.color in colored_by_agent:
                        colored_by_agent[obj.color] += 1
                    else:
                        colored_by_agent[obj.color] = 1

        print("colored tiles: ", colored_tiles,
              " agent contributions: ", colored_by_agent)
