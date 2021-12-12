from learning.utils.agent import Agent
import numpy as np
import torch
import datetime

from Coloring.learning.utils.arguments import vis_args
from Coloring.learning.utils.other import seed
from Coloring.learning.utils.storage import get_model_dir, save_capture
from Coloring.learning.utils.env import make_env

# Parse arguments
args = vis_args()

# Set seed for all randomness sources

seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = make_env(
    args.env, args.agents, grid_size=args.grid_size, agent_view_size=args.agent_view_size, setting=args.setting, market=args.market, seed=args.seed, max_steps=args.max_steps)

for _ in range(args.shift):
    env.reset()

print("Environment loaded\n")

# Load agent

model_dir = get_model_dir(args.model)
agents = []
if args.market:
    action_space = env.action_space.nvec.prod()
else:
    action_space = env.action_space.n
for agent in range(args.agents):
    agents.append(Agent(args.algo, agent, env.observation_space, action_space, model_dir,
                        device=device))
print("Agents loaded\n")

# Run the agent

if args.capture:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    print("Episode " + str(episode+1))
    obs = env.reset()

    while True:
        env.render('human')
        if args.capture:
            frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))
        joint_actions = np.array((1, len(agents)))
        for agent_index, agent in enumerate(agents):
            action = agent.get_action(obs, agent_index)
            joint_actions[agent_index] = action

        obs, reward, done, info = env.step(joint_actions)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.capture:
    now = datetime.datetime.now()
    name = datetime.datetime.strftime(now, '%d%m%Y') + "_" + datetime.datetime.strftime(now, '%H%M')
    save_capture(model_dir, name + ".gif", np.array(frames), pause=args.pause, training=False)
