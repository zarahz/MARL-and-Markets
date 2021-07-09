import argparse
import time
import numpy
import torch

from learning.utils import *


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--env", default='Empty-Grid-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--agents", default=1, type=int,
                    help="amount of agents")
parser.add_argument("--percentage-reward", default=False,
                    help="reward agents based on percentage of coloration in the grid (default: False)")
parser.add_argument("--mixed-motive", default=False,
                    help="If set to true the reward is not shared which enables a mixed motive environment (one vs. all). Otherwise agents need to work in cooperation to gain more reward. (default: False = Cooperation)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")

args = parser.parse_args()

# Set seed for all randomness sources

learning.utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = learning.utils.make_env(
    args.env, args.agents, args.seed, args.percentage_reward, args.mixed_motive)
# for _ in range(args.shift):
#     env.reset()
print("Environment loaded\n")

# Load agent

model_dir = learning.utils.get_model_dir(args.model)
agents = []
for agent in range(args.agents):
    agents.append(learning.utils.Agent(agent, env.observation_space, env.action_space, model_dir,
                                       device=device, argmax=args.argmax))
print("Agents loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        joint_actions = []
        for agent_index, agent in enumerate(agents):
            action = agent.get_action(obs, agent_index)
            joint_actions.append(action)

        obs, reward, done, _ = env.step(joint_actions)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
