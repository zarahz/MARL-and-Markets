import gym
from gym import spaces
from environment.colors import *

import environment


class CooperativeMultiagentWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, percentage_reward=False, mixed_motive=False, tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size
        self.percentage_reward = percentage_reward
        self.mixed_motive = mixed_motive

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        if done:  # adjust reward to be calculated out of percentage
            reward = self.calculate_reward()
        else:
            reward = [0]*len(self.env.agents)
        return observation, reward, done, info

    def calculate_reward(self):
        agents = self.env.agents
        if self.mixed_motive:
            reward = []  # reward of agent is its index
            # self.env.colored_cells() returns all cell encodings that contain a one in the middle
            # i.e. [3,1,2] -> 1 = cell is colored
            # with the indexing a new array is created that only contains the last value of the encoding (the color)
            cell_colors = self.env.colored_cells()[:, 2]
            for agent in range(len(agents)):
                agent_coloration = (
                    cell_colors == self.env.agents[agent]['color']).sum()
                color_percentage = agent_coloration / \
                    len(self.env.walkable_cells())
                reward.append(color_percentage)
            print(reward)
            return reward
        else:
            # coop reward based on coloring percentage
            if self.percentage_reward:
                return [1 * self.env.grid_colored_percentage()]*len(agents)

            # coop reward based on completed coloring
            if self.env.whole_grid_colored():
                print(self.env.whole_grid_colored(),
                      '---- GRID FULLY COLORED! ----')
                return [1]*len(agents)

            return [0]*len(agents)
