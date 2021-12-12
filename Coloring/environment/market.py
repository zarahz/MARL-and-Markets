
import numpy as np
import torch
import copy


class Market:

    trading_matrix = None

    def __init__(
        self,
        type,
        trading_fee,
        agents
    ):
        # the market logic is either action (am) or shareholder (sm) market
        assert "am" in type or "sm" in type, "Market Type is either Shareholder (sm) or Action (am)"
        self.type = type
        self.trading_fee = trading_fee
        self.agents = agents

    def execute_market_actions(self, env_actions, market_actions, env_reward, done, reset_fields_by=[]):
        trading_info = {"trades": 0}

        # if not is_last_step:
        if "sm" in self.type:
            trading_info["trades"], trading_reward = self.execute_sm(
                market_actions, env_reward, reset_fields_by)
        else:
            trading_info["trades"], trading_reward = self.execute_am(
                market_actions, env_actions, env_reward, reset_fields_by)

        reward = [r + tr for r, tr in zip(env_reward, trading_reward)]

        # if done then the final reward calculation would trigger that function
        if not done and "sm" in self.type:
            reward = self.calculate_traded_reward(reward)
        return trading_info, reward
        
    def execute_sm(self, market_actions, env_reward, reset_fields_by):
        shares = torch.diagonal(self.trading_matrix, 0)
        trades = 0
        rewards = [0]*self.agents
        price = 0  # self.trading_fee/2
        buying_matrix, selling_matrix = self.extract_selling_buying_matrices(
            market_actions)
        for seller, selling_row in enumerate(selling_matrix):
            for buyer, buying_row in enumerate(buying_matrix):
                if not selling_row[seller] or seller == buyer:
                    continue

                if self.reset_fields(reset_fields_by, buyer) or (price > 0 and self.debt(price, env_reward[buyer])):
                    continue

                # TODO shuffel buying_agents for fairness
                if(torch.eq(selling_row, buying_row).all() and shares[seller] > self.trading_fee):
                    trades += 1
                    # in case of share exchange fixed price of a third trading fee!
                    rewards = self.update_rewards(
                        rewards, buyer, seller, price)

                    # trading fee is the share so the buying agent gets the share of the selling agent here!
                    self.update_trading_matrix(buyer, seller)
        return trades, rewards

    def execute_am(self, market_actions, env_actions, env_reward, reset_fields_by):
        trades = 0
        rewards = [0]*self.agents

        for buyer, offer in enumerate(market_actions):
            buy_from = offer[0]

            if self.reset_fields(reset_fields_by, buy_from) or self.debt(self.trading_fee, env_reward[buyer]):
                continue

            if self.not_waiting_or_acting_on_self(buy_from, buyer) and env_actions[buy_from] == offer[1]:
                trades += 1

                if "am-goal" not in self.type:
                    # only in case of plain "am" a reward is directly calculated, else a condition must be met
                    # i.e. for "am-goal" the goal state must be fullfilled
                    rewards = self.update_rewards(
                        rewards, buyer, buy_from, self.trading_fee)
                self.update_trading_matrix(buy_from, buyer)

        return trades, rewards

    def update_rewards(self, rewards, reduce_from, add_to, price):
        rewards[reduce_from] -= price
        rewards[add_to] += price
        return rewards

    def update_trading_matrix(self, reciever, buyer):
        # trading fee is the share so the buying agent gets the share of the selling agent here!
        self.trading_matrix[reciever][buyer] += self.trading_fee
        self.trading_matrix[buyer][buyer] -= self.trading_fee

    def reset_fields(self, agents_that_reset_fields, reciever):
        '''
        prevent agents to be rewarded/traded with if it reset fields in the current step
        (only applied when market type contains setting "no-reset")
        '''
        return "no-reset" in self.type and reciever in agents_that_reset_fields

    def debt(self, price, balance):
        '''
        checks market setting and returns boolean  
        False -> the price is in the budget otherwise true for debt!
        '''
        return "no-debt" in self.type and price <= balance

    def not_waiting_or_acting_on_self(self, recierver, actor):
        '''
        check if the agent (actor) is trying to trade with itself and also
        checking if the reciever is in agent count, else the agent does not execute a market action but waits

        '''
        return recierver <= self.agents-1 and recierver != actor

    def extract_selling_buying_matrices(self, market_actions):
        # each matrix row presents agent -> if row value is one then index is interaction with agent[index]
        buying_matrix = torch.zeros(
            (self.agents, self.agents), dtype=torch.int)
        selling_matrix = torch.zeros(
            (self.agents, self.agents), dtype=torch.int)

        for agent, offer in enumerate(market_actions):
            buy_from = offer[0]
            selling = offer[1]
            # if buy_from is greater than agents-1 then do nothing
            if self.not_waiting_or_acting_on_self(buy_from, agent):
                buying_matrix[agent][buy_from] = 1
            if selling:
                selling_matrix[agent][agent] = 1

        return buying_matrix, selling_matrix

    def calculate_traded_reward(self, rewards, env_goal=False):
        if ("am" in self.type and not "goal" in self.type) or ("goal" in self.type and not env_goal):
            # in this case no market trades will be executed since goal was set but not reached
            # or normal action market was already executed
            return rewards

        trading_rewards = [0]*len(rewards)

        for agent in range(len(rewards)):
            if "am" in self.type:
                # only in this am scenario change rewards at the end
                trading_rewards[agent] = rewards[agent] + \
                    sum(self.trading_matrix[agent])
            else:  # shareholder market
                # iterate all trading matrix rows and env rewards
                for index, (trade, reward) in enumerate(zip(self.trading_matrix[agent], rewards)):
                    # agents are in debt?
                    if reward <= 0:
                        if index != agent:
                            # in this case trading agent has debt so do nothing!
                            continue
                        else:
                            # here the receiving agent has debt so that should stay
                            trading_rewards[agent] += reward
                    else:
                        trading_rewards[agent] += (trade * reward).item()
        return trading_rewards

    def reset(self):
        self.trading_matrix = torch.zeros(
            (self.agents, self.agents), dtype=torch.float)
        if "sm" in self.type:
            # fill trading matrix diagonal with ones as overall share of each agent
            self.trading_matrix.fill_diagonal_(1, wrap=False)
