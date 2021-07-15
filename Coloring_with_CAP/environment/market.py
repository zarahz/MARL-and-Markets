import numpy as np


class Market:

    trading_fee = 0.05
    balance = []

    def __init__(
        self,
        type
    ):
        # the market logic is either action (am) or shareholder (sm) market
        assert type == "am" or type == "sm"
        self.type = type

    def match_actions(self, executed_actions, market_actions):
        for agent, action in enumerate(executed_actions):
            for trading_agent, trading_action in enumerate(market_actions):
                if agent != trading_agent and action == trading_action and self.balance[trading_agent] > self.trading_fee:
                    self.balance[trading_agent] -= self.trading_fee
                    self.balance[agent] += self.trading_fee
                    print("balance changed to: ", self.balance)

    def calculate_traded_reward(self, rewards):
        trading_rewards = [reward*trade for reward,
                           trade in zip(rewards, self.balance)]
        return trading_rewards

    def reset(self, agents):
        if self.type == "sm":
            self.balance = [1/agents]*agents
        else:  # am
            self.balance = [0]*agents  # no prices on init
