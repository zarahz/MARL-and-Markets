
import numpy as np
import torch


class Market:

    trading_matrix = None

    def __init__(
        self,
        type
    ):
        # the market logic is either action (am) or shareholder (sm) market
        assert type == "am" or type == "sm", "Market Type is either Shareholder (sm) or Action (am)"
        self.type = type

    def execute_market_actions(self, market_actions, trading_fee, is_last_step=False):
        agents = market_actions.shape[0]

        trading_info = {"trades": 0}
        reward = [0]*agents

        # if not is_last_step:
        trading_info, reward = self.update_trading_matrix(
            agents, market_actions, trading_fee, trading_info, reward)

        return trading_info, reward
        # matrix cols need to equal 1, since total share is 100%
        # TODO float error 0.9999999999..
        # assert all(total_shares == 1 for total_shares in torch.sum(self.trading_matrix, dim=0).tolist())

    def update_trading_matrix(self, agents, market_actions, trading_fee, trading_info, reward):
        balance = torch.diagonal(self.trading_matrix, 0)
        buying_matrix, selling_matrix = self.extract_selling_buying_matrices(agents,
                                                                             market_actions)
        for selling_agent, selling_row in enumerate(selling_matrix):
            for buying_agent, buying_row in enumerate(buying_matrix):
                if not selling_row[selling_agent] or selling_agent == buying_agent:
                    continue
                # TODO shuffel buying_agents for fairness
                if(torch.eq(selling_row, buying_row).all() and balance[selling_agent] >= trading_fee):
                    trading_info["trades"] += 1

                    # in case of share exchange fixed price!
                    reward[buying_agent] -= (trading_fee/3)
                    reward[selling_agent] += (trading_fee/3)

                    # and save up price
                    self.trading_matrix[buying_agent][selling_agent] += trading_fee
                    self.trading_matrix[selling_agent][selling_agent] -= trading_fee

        return trading_info, reward

    def extract_selling_buying_matrices(self, agents, market_actions):
        # each matrix row presents agent -> if row value is one then index is interaction with agent[index]
        buying_matrix = torch.zeros((agents, agents), dtype=torch.int)
        selling_matrix = torch.zeros((agents, agents), dtype=torch.int)

        for agent, offer in enumerate(market_actions):
            buy_from = offer[0]
            selling = offer[1]
            # if buy_from is greater than agents-1 then do nothing
            if buy_from != agent and buy_from <= agents-1:
                buying_matrix[agent][buy_from] = 1
            if selling:
                selling_matrix[agent][agent] = 1

        return buying_matrix, selling_matrix

    def calculate_traded_reward(self, rewards):
        trading_rewards = [0]*len(rewards)
        for agent in range(len(rewards)):
            # in case of sm -> trade only offers positive reward
            for a, share in enumerate(self.trading_matrix[agent]):
                if(rewards[a] > 0):
                    trading_rewards[agent] += (rewards[a] * share).item()
        return trading_rewards

    def reset(self, agents):
        if self.type == "sm":
            # 100% shares
            self.trading_matrix = torch.zeros(
                (agents, agents), dtype=torch.float)
            # fill trading matrix diagonal with ones as overall share of each agent
            self.trading_matrix.fill_diagonal_(1, wrap=False)
        else:
            self.balance = [0]*agents
