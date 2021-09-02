import gym
import numpy as np
import environment.market as market


class MultiagentWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, percentage_reward=False, mixed_motive=False, tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size
        self.percentage_reward = percentage_reward
        self.mixed_motive = mixed_motive
        if self.env.market:
            self.market = market.Market(self.env.market, self.env.trading_fee)

    def reset(self):
        if self.market:
            self.market.reset(len(self.env.agents))
        observation = self.env.reset()
        return observation

    def step(self, actions):
        if self.market:
            # ------------------------------------------------------------
            actions = self.env.decode_actions(actions)  # TODO! uncomment
            # ------------------------------------------------------------
            # always take the first action, since the following are only relevant for the market
            market_actions = actions[:, 1:]
            actions = actions[:, 0]
            # is_last_step = (self.env.step_count+1 >= self.env.max_steps)
            trades, trading_reward = self.market.execute_market_actions(actions,
                                                                        market_actions)
        observation, reward, done, info = self.env.step(actions)

        # reward is an array of length agents
        # array formation, so that mixed motive rewards are easily adapted (each agent is rewarded seperately)
        reward = [0]*len(self.env.agents)

        if(self.market):
            info.update(trades)
            reward = trading_reward
        if done:
            reward = self.calculate_reward(reward)

        return observation, reward, done, info

    def calculate_reward(self, reward):
        agents = self.env.agents
        if self.mixed_motive:
            # self.env.colored_cells() returns all cell encodings that contain a one in the middle
            # i.e. [3,1,2] -> 1 = cell is colored
            # with the indexing a new array is created that only contains the last value of the encoding (the color)
            cell_colors = self.env.colored_cells()[:, 2]
            for agent in range(len(agents)):
                # reward agent for the percentage of its coloration on the field
                agent_coloration = (
                    cell_colors == self.env.agents[agent]['color']).sum()
                color_percentage = agent_coloration / \
                    len(self.env.walkable_cells())
                reward[agent] += color_percentage
        else:
            # coop reward based on completed coloring
            if self.env.whole_grid_colored():
                print('---- GRID FULLY COLORED! ---- steps', self.env.step_count)
                reward = [r + one for r, one in zip(reward, [1]*len(agents))]

            # coop reward based on coloring percentage
            elif self.percentage_reward:
                percentage_reward = [
                    1 * self.env.grid_colored_percentage()]*len(agents)
                reward = [r + percentage for r,
                          percentage in zip(reward, percentage_reward)]

        # execute market calculations too
        if self.market:
            reward = self.market.calculate_traded_reward(reward)
        return reward
