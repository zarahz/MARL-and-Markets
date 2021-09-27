
import numpy
import torch

from Coloring.learning.ppo.utils import *

from Coloring.learning.utils.storage import get_model_dir
from Coloring.learning.utils.arguments import vis_args
from Coloring.learning.utils.env import make_env

# Set device
args = vis_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = make_env(
    args.env, args.agents, grid_size=args.grid_size, setting=args.setting, seed=args.seed)
# for _ in range(args.shift):
#     env.reset()
print("Environment loaded\n")

# Load agent

model_dir = get_model_dir(args.model)
agents = []
if args.market:
    action_space = env.action_space.nvec.prod()
else:
    action_space = env.action_space.n
for agent in range(args.agents):
    # TODO implement agent logic for dqn!
    agents.append(Agent(agent, env.observation_space, action_space, model_dir,
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
