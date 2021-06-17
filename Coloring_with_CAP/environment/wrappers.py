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
        if hasattr(self, 'grid'):
            self.print_coloring_data()

        observation = self.env.reset()
        return observation

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        if done:
            reward = self.calculate_reward()
        return observation, reward, done, info

    def calculate_reward(self):
        if self.env.whole_grid_colored():
            print('---- GRID FULLY COLORED! ----')
            return 10
        return 0

    def print_coloring_data(self):
        floor_tiles = 0
        colored_tiles = 0
        colored_by_agent = {}
        for obj in self.env.grid.grid:
            if isinstance(obj, environment.grid.Floor):
                floor_tiles += 1
                if obj.status == 'colored':
                    colored_tiles += 1
                    if obj.color in colored_by_agent:
                        colored_by_agent[obj.color] += 1
                    else:
                        colored_by_agent[obj.color] = 1

        print("walkable Floor tiles: ", floor_tiles)
        print("floor tiles that are colored: ", colored_tiles)
        print("agent contributions:", colored_by_agent)
