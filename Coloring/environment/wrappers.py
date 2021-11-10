import copy
import gym
import numpy as np
import environment.market as market


class MultiagentWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, setting="", trading_fee=0.1, tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size
        self.setting = setting
        self.upper_step_reward_bound = 0.1
        self.lower_step_reward_bound = -0.1
        if self.env.market:
            self.market = market.Market(
                self.env.market, trading_fee, len(self.env.agents))

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

        if "difference-reward" in self.setting:
            reward_sum = sum(reward)
            for agent in range(len(self.env.agents)):
                temp_reward = copy.deepcopy(reward)
                temp_reward[agent] = 0 # zero since dr is calculated with waiting action
                temp_reward_sum = sum(temp_reward)
                dr = reward_sum - temp_reward_sum

                reward[agent] = self.clip_coop_rewards(dr)[agent]
        elif not "mixed" in self.setting: # normal coop mode
            # assign all agents the same reward since here the reward is containig positive values for agents
            # that have colored a field!
            coop_reward = sum(reward)
            reward = self.clip_coop_rewards(coop_reward)

        if(self.market):
            # is_last_step = (self.env.step_count+1 >= self.env.max_steps)
            trades, reward = self.market.execute_market_actions(actions,
                                                                market_actions, reward, done, info["reset_fields_by"])
            info.update(trades)

        if done:
            reward = self.calculate_reward(reward, info)

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
            percentage_reward = [
                1 * self.env.grid_colored_percentage()]*len(agents)
            reward = [r + percentage for r,
                      percentage in zip(reward, percentage_reward)]

            # coop reward with difference calculation to solve CAP
            if "difference-reward" in self.setting:
                for agent in range(len(agents)):
                    reward[agent] = reward[agent] - info["difference_rewards"][agent]

        # execute market calculations too
        if self.market:
            reward = self.market.calculate_traded_reward(
                reward, env_goal_reached)
        return reward
    
    def clip_coop_rewards(self, rewards):
        # in coop case prevent reward getting too big/small since the reward 
        # sum is taken 
        if rewards >= self.upper_step_reward_bound:
            reward = [self.upper_step_reward_bound]*len(self.env.agents)
        elif rewards <= self.lower_step_reward_bound:
            reward = [self.lower_step_reward_bound]*len(self.env.agents)
        else:
            reward = [rewards]*len(self.env.agents)
        return reward
