import gym
from environment.wrappers import *
from environment.window import Window
from environment.colors import *
import argparse
import numpy as np
import math

from Coloring.learning.ppo.utils.arguments import get_train_args

# --------- Settings
args = get_train_args()
# --------- Hyperparams
epsilon = 0.2  # 0.01
tau = 0.75
agents = args.agents


def redraw():
    img = env.render('rgb_array', tile_size=32)
    window.show_img(img)


def reset():
    print("resetted")
    env.reset()
    # if hasattr(env, 'mission'):
    #     window.set_caption(env.mission)
    redraw()


def visualize(done):
    if done:
        print("done")
        reset()
    else:
        redraw()


def epsilon_greedy():
    # rand = np.random.random()
    # if rand < epsilon:
    action = env.action_space.sample()
    # else:
    #     action = np.argmax(Q)
    return action


def softmax():
    # sum over all actions
    sum = 0

    # probability per action
    policy = []

    for q in Q:
        sum = sum + math.exp(q/tau)
    for q in Q:
        policy.append(math.exp(q/tau) / sum)

    return np.random.choice(len(Q), 1, p=policy)[0]


#######################
# conduct experiment
#######################
market = args.market  # "sm-no-reset-goal"
trading_fee = 0.05
competitive = "competitive" in args.setting
# env = gym.make(id=args.env, agents=args.agents,
#                agent_view_size=args.agent_view_size, max_steps=args.max_steps, competitive=competitive, market=market, trading_fee=trading_fee, size=args.size)
env = gym.make(id=args.env, agents=args.agents, size=args.grid_size, competitive=competitive, agent_view_size=args.agent_view_size,
               market=args.market, trading_fee=args.trading_fee, max_steps=args.max_steps)
# wrapper for environment adjustments
env = MultiagentWrapper(env, args.setting)
env.seed(args.seed)

window = Window(args.env)


action_space = env.action_space.nvec[0] if market else env.action_space.n
if "mixed-motive" in args.setting:
    # variable to count the number of times an action was pulled
    action_count = [np.zeros(action_space)]*agents
    # Q value => expected average reward
    Q = [np.zeros(len(action_count))]*agents
else:
    action_count = np.zeros(action_space)
    Q = np.zeros(len(action_count))

reset()
for episode in range(100):
    s = 0
    for s in range(args.max_steps):
        joint_actions = []
        for agent in range(agents):
            # select the action (1,1,2)
            action = epsilon_greedy()  # softmax()
            # update the count of that action
            if "mixed-motive" in args.setting:
                action_count[agent][action] += 1
            else:
                action_count[action] += 1
            joint_actions.append(action)

        # action: left = 1 | right = 2 | up = 3 | down = 4 | wait = 0

        # get reward/observation/terminalInfo
        observation, reward, done, info = env.step(np.array(joint_actions))
        print(*reward, sep=", ")
        print(info)

        # if "mixed-motive" in args.setting:
        #     for agent in range(agents):
        #         # recalculate its Q value
        #         Q[agent][action] = Q[agent][action] + (1/action_count[agent][action]) * \
        #             (reward[agent]-Q[agent][action])
        # else:
        #     Q[action] = Q[action] + (1/action_count[action]) * \
        #         (reward[0]-Q[action])
        visualize(done)
        if done:
            print('done! step=', s, ' reward=', reward, ' info=', info)

window.close()
