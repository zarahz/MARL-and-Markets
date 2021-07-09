import argparse
import time
import gym
import torch
from environment.wrappers import CooperativeMultiagentWrapper

import learning.utils
from matplotlib import pyplot as plt


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--agents", default=1, type=int,
                    help="amount of agents")
parser.add_argument("--episodes", type=int, default=30,
                    help="number of episodes of evaluation (default: 30)")
parser.add_argument("--max-steps", type=int, default=100,
                    help="max number of steps in episodes to reach the goal (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
# parser.add_argument("--procs", type=int, default=16,
#                     help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
args = parser.parse_args()

# Set seed for all randomness sources

learning.utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments
# print("NAME:________________________  ", __name__)
# if __name__ == '__main__':
# env = learning.utils.make_env(
#     args.env, args.agents, args.seed + 10000)
# env = ParallelEnv([env])
env = gym.make(id=args.env, agents=args.agents, max_steps=args.max_steps)
env = CooperativeMultiagentWrapper(env)
env.seed(args.seed + 10000)
print("Environments loaded\n")

# Load agent

model_dir = learning.utils.get_model_dir(args.model)
agents = []
for agent in range(args.agents):
    agents.append(learning.utils.Agent(agent, env.observation_space, env.action_space, model_dir,
                                       device=device))
print("Agent loaded\n")

# Initialize logs

logs = {"reward_per_episode": [], "resetted_fields_per_episode": []}

for agent in range(len(agents)):
    # needed to plot shared and independant rewards
    key = "reward_of_agent_"+agent
    logs[key] = []

# Run agent

start_time = time.time()

obs = env.reset()

# log_done_counter = 0
# return_per_episode = torch.zeros(args.episodes, device=device)
# log_episode_num_frames = torch.zeros(args.episodes, device=device)

for episode in range(args.episodes):

    for step in range(args.max_steps):
        joint_actions = []
        for agent_index, agent in enumerate(agents):
            action = agent.get_actions(obs, agent_index)
            joint_actions.append(action)

        obs, reward, done, info = env.step(joint_actions)
        info["resetted_fields"] = step
        logs["reward_per_episode"][episode] += reward
        logs["resetted_fields_per_episode"][episode] += info["resetted_fields"]

end_time = time.time()

# Plot
for key, value in logs.items():
    plt.plot(value)
    plt.show()

    # num_frames = sum(logs["num_frames_per_episode"])
    # fps = num_frames/(end_time - start_time)
    # duration = int(end_time - start_time)
    # return_per_episode = learning.utils.synthesize(logs["return_per_episode"])
    # num_frames_per_episode = learning.utils.synthesize(
    #     logs["num_frames_per_episode"])

    # print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
    #       .format(num_frames, fps, duration,
    #               *return_per_episode.values(),
    #               *num_frames_per_episode.values()))

    # # Print worst episodes

    # n = args.worst_episodes_to_show
    # if n > 0:
    #     print("\n{} worst episodes:".format(n))

    #     indexes = sorted(range(
    #         len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    #     for i in indexes[:n]:
    #         print("- episode {}: R={}, F={}".format(i,
    #               logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
