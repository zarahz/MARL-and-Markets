import gym
from environment.wrappers import *
from environment.window import Window
from environment.colors import *
import argparse
import numpy as np
import math

# --------- Settings
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment(REQUIRED)")
parser.add_argument("--agents", required=True, type=int,
                    help="number of agents (REQUIRED)")
parser.add_argument("--mixed-motive", default=False,
                    help="If set to true the reward is not shared which enables a mixed motive environment (one vs. all). Otherwise agents need to work in cooperation to gain more reward. (default: False = Cooperation)")
# optional
parser.add_argument("--agent_view_size", type=int, default=5,
                    help="partial view size of the agents, needs to be an odd number! (default: 5)")
parser.add_argument("--max_steps", type=int, default=10,
                    help="max. steps of the agents per episode (default: 100)")
parser.add_argument("--episodes", type=int, default=100,
                    help="iterations of the game (default: 100)")
parser.add_argument("--size", type=int, default=8,
                    help="size of the grid (default: 8)")
args = parser.parse_args()
# --------- Hyperparams
epsilon = 0.2  # 0.01
tau = 0.75
agents = args.agents


def redraw():
    img = env.render('rgb_array', tile_size=32)
    window.show_img(img)


def reset():
    env.reset()
    if hasattr(env, 'mission'):
        window.set_caption(env.mission)
    redraw()


def visualize(done):
    if done:
        reset()
    else:
        redraw()


def epsilon_greedy():
    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
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
env = gym.make(id=args.env, agents=args.agents,
               agent_view_size=args.agent_view_size, max_steps=args.max_steps, size=args.size)
env = CooperativeMultiagentWrapper(env)  # wrapper for environment adjustments

window = Window(args.env)


if args.mixed_motive:
    # variable to count the number of times an action was pulled
    action_count = [np.zeros(env.action_space.n)]*agents
    # Q value => expected average reward
    Q = [np.zeros(len(action_count))]*agents
else:
    action_count = np.zeros(env.action_space.n)
    Q = np.zeros(len(action_count))
reset()
for episode in range(args.episodes):
    # s = 0
    for s in range(args.max_steps):
        joint_actions = []
        for agent in range(agents):
            # select the action (1,1,2)
            action = epsilon_greedy()  # softmax()
            # update the count of that action
            if args.mixed_motive:
                action_count[agent][action] += 1
            else:
                action_count[action] += 1
            joint_actions.append(action)

        # get reward/observation/terminalInfo
        observation, reward, done, info = env.step(joint_actions)

        if args.mixed_motive:
            for agent in range(agents):
                # recalculate its Q value
                Q[agent][action] = Q[agent][action] + (1/action_count[agent][action]) * \
                    (reward[agent]-Q[agent][action])
        else:
            Q[action] = Q[action] + (1/action_count[action]) * \
                (reward[0]-Q[action])
        print(reward)
        visualize(done)
        if done:
            print('done! step=', s, ' reward=', reward)

window.close()
