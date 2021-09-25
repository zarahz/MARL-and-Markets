import numpy as np
import torch

from learning.ppo.utils import *
from learning.ppo.utils.arguments import get_vis_args
from learning.utils.other import seed


# Parse arguments
args = get_vis_args()

# Set seed for all randomness sources

seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = learning.utils.make_env(
    args.env, args.agents, grid_size=args.grid_size, agent_view_size=args.agent_view_size, setting=args.setting, market=args.market, seed=args.seed)
# for _ in range(args.shift):
#     env.reset()
print("Environment loaded\n")

# Load agent

model_dir = learning.utils.get_model_dir(args.model)
agents = []
if args.market:
    action_space = env.action_space.nvec.prod()
else:
    action_space = env.action_space.n
for agent in range(args.agents):
    agents.append(learning.ppo.utils.Agent(agent, env.observation_space, action_space, model_dir,
                                           device=device, argmax=args.argmax))
print("Agents loaded\n")

# Run the agent

if args.capture:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render('human')
        if args.capture:
            frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))
        joint_actions = np.array((1, len(agents)))  # []
        for agent_index, agent in enumerate(agents):
            action = agent.get_action(obs, agent_index)
            # joint_actions.append(action)
            joint_actions[agent_index] = action

        obs, reward, done, info = env.step(joint_actions)
        print(info)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.capture:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")