import gym
import numpy as np
import environment.market as market


class MultiagentWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, setting="", tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size
        self.setting = setting
        if self.env.market:
            self.market = market.Market(
                self.env.market, self.env.trading_fee, len(self.env.agents))

    def reset(self):
        if self.market:
            self.market.reset()
        observation = self.env.reset()
        return observation

    def step(self, actions):
        if self.market:
            # ------------------------------------------------------------
            actions = self.env.decode_actions(actions)  # TODO! uncomment
            # ------------------------------------------------------------
            # split environment actions from market actions
            # i.e. agent 1 action: [env_action, market_action1, market_action2]
            market_actions = actions[:, 1:]
            actions = actions[:, 0]

        observation, reward, done, info = self.env.step(
            actions, "difference-reward" in self.setting)

        if not "mixed" in self.setting:
            # assign all agents the same reward since here the reward is containig positive values for agents
            # that have colored a field!
            coop_reward = sum(reward)

            # in coop case prevent reward getting too big/small
            if coop_reward >= 0.1:
                reward = [0.1]*len(self.env.agents)
            elif coop_reward <= -0.1:
                reward = [-0.1]*len(self.env.agents)
            else:
                reward = [coop_reward]*len(self.env.agents)

        if(self.market):
            # is_last_step = (self.env.step_count+1 >= self.env.max_steps)
            trades, trading_reward = self.market.execute_market_actions(actions,
                                                                        market_actions, reward, info["reset_fields_by"])
            info.update(trades)
            reward = [r + tr for r, tr in zip(reward, trading_reward)]

        if done:
            reward = self.calculate_reward(reward, info)
        else:
            # prevent applying dr twice
            self.calculate_difference_reward(reward, info)

        return observation, reward, done, info

    def calculate_reward(self, reward, info):
        agents = self.env.agents
        env_goal_reached = self.env.whole_grid_colored(self.env.grid)
        if env_goal_reached:
            print('---- GRID FULLY COLORED! ---- steps', self.env.step_count)

        if "mixed-motive" in self.setting:
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
            if env_goal_reached:
                reward = [r + one for r, one in zip(reward, [1]*len(agents))]

            # coop reward with difference calculation to solve CAP
            reward = self.calculate_difference_reward(
                reward, info)

        # execute market calculations too
        if self.market:
            reward = self.market.calculate_traded_reward(
                reward, env_goal_reached)
        return reward

    def calculate_difference_reward(self, reward, info):
        if "difference-reward" in self.setting:
            for agent, agent_reward in enumerate(reward):
                default_action_reward = info['difference_reward'][agent]['reward']
                if info['difference_reward'][agent]["fully_colored"]:
                    # harsh penalty if environment would have been colored! otherwise just subtract a smaller value
                    goal_reward = 1  # if env_colored == False else 0.5
                    default_action_reward = [r + goal_r for r,
                                             goal_r in zip(default_action_reward, [goal_reward]*len(reward))]
                reward[agent] = agent_reward - default_action_reward[agent]
        return reward
