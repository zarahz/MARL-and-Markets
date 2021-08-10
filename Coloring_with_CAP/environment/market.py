
class Market:

    balance = []

    def __init__(
        self,
        type
    ):
        # the market logic is either action (am) or shareholder (sm) market
        assert type == "am" or type == "sm", "Market Type is either Shareholder (sm) or Action (am)"
        self.type = type

    def calculate_balance(self, executed_actions, market_info, trading_fee):
        for agent, action in enumerate(executed_actions):
            for trading_agent, trade_condition in enumerate(market_info):
                execute_trade = agent != trading_agent and self.action_match(
                    agent, trading_agent, action, trade_condition, trading_fee)
                if execute_trade:
                    self.balance[trading_agent] -= trading_fee
                    self.balance[agent] += trading_fee
                    print("balance changed to: ", self.balance)

    def action_match(self, agent, trading_agent, action, trade_condition, trading_fee):
        '''
        Here the market type plays a role. The Shareholder maket is unconditional and only checks if actions match
        whereas the Action market matches the recieving agent with the trading condition as well as its executed action.
        '''
        if self.type == "sm":
            return action == trade_condition and self.balance[trading_agent] >= trading_fee
        else:
            # prevent dept exceeding total reward # TODO necessary?
            below_max_dept = sum(self.balance) >= -1
            # TODO total reward (1) is hardcoded here!
            return agent == trade_condition[0] and action == trade_condition[1] and below_max_dept

    def calculate_traded_reward(self, rewards):
        trading_rewards = []
        for reward, trade in zip(rewards, self.balance):
            if self.type == "sm":
                # percentage based
                trading_rewards.append(reward*trade)
            else:
                # price based
                trading_rewards.append(reward+trade)
        return trading_rewards

    def reset(self, agents):
        if self.type == "sm":
            self.balance = [1/agents]*agents
        else:
            self.balance = [0]*agents
