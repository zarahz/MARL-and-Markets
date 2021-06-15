import gym
from environment.wrappers import *
from environment.window import Window
from environment.colors import *
import argparse
import numpy as np
import math

# --------- Settings
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment(REQUIRED)")
parser.add_argument("--agents", required=True, type=int,
                    help="number of agents (REQUIRED)")
# optional
parser.add_argument("--agent_view_size", type=int, default=5,
                    help="partial view size of the agents (default: 5)")
parser.add_argument("--max_steps", type=int, default=100,
                    help="max. steps of the agents per episode (default: 100)")
parser.add_argument("--episodes", type=int, default=100,
                    help="iterations of the game (default: 100)")
parser.add_argument("--size", type=int, default=8,
                    help="size of the grid (default: 8)")
args = parser.parse_args()
# --------- Hyperparams
# settings
# episodes = 10
epsilon = 0.2  # 0.01
tau = 0.75

# layout = 'Empty-Grid-8x8-v0'

agents = args.agents


def interpret_action(action):
    action_string = {
        0: "left",
        1: "right",
        2: "up",
        3: "down",
        4: "wait"
    }
    return action_string.get(action, "")


def redraw():
    img = env.render('rgb_array', tile_size=32)
    window.show_img(img)


def reset():
    # reset for each agent
    env.reset()
    if hasattr(env, 'mission'):
        window.set_caption(env.mission)
    redraw()


def step(done):
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
env = MultiagentFullyObsWrapper(env)  # Get pixel observations

window = Window(args.env)


# count of number of times an action was pulled
# aktion ist zB (U,-),(U,U),... U=up
# trade abbilden dh erhöhen, um alle trades zu involvieren
action_count = np.zeros(env.action_space.n)
# Q value => expected average reward
# Q = np.loadtxt('q_learning.txt')  # np.zeros(len(action_count))
# if Q.size == 0:
Q = np.zeros(len(action_count))
for episode in range(args.episodes):
    reset()  # This now produces an RGB tensor only
    s = 0
    for s in range(args.max_steps):
        joint_actions = []
        # market noch mit reinbringen um aktionen zu traden und rewards zu verteilen, siehe paper
        for agent in range(agents):
            # select the action using epsilon greedy
            action = softmax()  # c
            joint_actions.append(action)
        # get reward/observation/terminalInfo
        observation, reward, done, info = env.step(joint_actions)
        # plt.imshow(observation['image'])
        # update the count of that action
        action_count[action] += 1
        # recalculate its Q value
        Q[action] = Q[action] + (1/action_count[action]) * \
            (reward-Q[action])

        step(done)
        if done:
            # np.savetxt('q_learning.txt', Q)
            if reward > 0:
                print('---- GRID FULLY COLORED! ----')
            print('done! step=%s, reward=%.2f' %
                  (s, reward))

window.close()
