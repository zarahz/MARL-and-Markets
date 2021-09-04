
import numpy as np
import torch


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

    def execute_market_actions(self, env_actions, market_actions, reset_fields_by=[], is_last_step=False):
        trading_info = {"trades": 0}

        # if not is_last_step:
        if "sm" in self.type:
            trading_info["trades"], reward = self.execute_sm(
                market_actions, reset_fields_by)
        else:
            trading_info["trades"], reward = self.execute_am(
                market_actions, env_actions, reset_fields_by)

        return trading_info, reward
        # matrix cols need to equal 1, since total share is 100%
        # TODO float error 0.9999999999..
        # assert all(total_shares == 1 for total_shares in torch.sum(self.trading_matrix, dim=0).tolist())

    def execute_sm(self, market_actions, reset_fields_by):
        shares = torch.diagonal(self.trading_matrix, 0)
        trades = 0
        rewards = [0]*self.agents
        buying_matrix, selling_matrix = self.extract_selling_buying_matrices(
            market_actions)
        for selling_agent, selling_row in enumerate(selling_matrix):
            for buying_agent, buying_row in enumerate(buying_matrix):
                if not selling_row[selling_agent] or selling_agent == buying_agent:
                    continue

                if self.apply_no_reset_market(reset_fields_by, buying_agent):
                    continue

                # TODO shuffel buying_agents for fairness
                if(torch.eq(selling_row, buying_row).all() and shares[selling_agent] >= self.trading_fee):
                    trades += 1
                    # in case of share exchange fixed price of a third trading fee!
                    rewards = self.update_rewards(
                        rewards, buying_agent, selling_agent, self.trading_fee/3)

                    # trading fee is the share so the buying agent gets the share of the selling agent here!
                    self.update_trading_matrix(buying_agent, selling_agent)
        return trades, rewards

    def execute_am(self, market_actions, env_actions, reset_fields_by):
        trades = 0
        rewards = [0]*self.agents

        for buyer, offer in enumerate(market_actions):
            buy_from = offer[0]

            if self.apply_no_reset_market(reset_fields_by, buy_from):
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

    def apply_no_reset_market(self, agents_that_reset_fields, reciever):
        '''
        prevent agents to be rewarded/traded with if it reset fields in the current step 
        (only applied when market type contains setting "no-reset")
        '''
        return "no-reset" in self.type and reciever in agents_that_reset_fields

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

    def calculate_traded_reward(self, rewards, env_goal):
        trading_rewards = [0]*len(rewards)
        for agent in range(len(rewards)):
            for a, trade in enumerate(self.trading_matrix[agent]):
                if ("goal" in self.type and not env_goal) or "am" in self.type:
                    # in this case either no market trades will be executed since goal is not reached
                    # or am market is the current type which always exchange rewards immediatly
                    return rewards

                if("sm" in self.type and rewards[a] > 0):
                    # only share positive reward with buyers
                    trading_rewards[agent] += (rewards[a] * trade).item()

                # elif("am" in self.type):
                #     trading_rewards[agent] += (rewards[a] + trade).item()

        return trading_rewards

    def reset(self):
        self.trading_matrix = torch.zeros(
            (self.agents, self.agents), dtype=torch.float)
        if "sm" in self.type:
            # fill trading matrix diagonal with ones as overall share of each agent
            self.trading_matrix.fill_diagonal_(1, wrap=False)
        # else:
        #     self.balance = [0]*agents
