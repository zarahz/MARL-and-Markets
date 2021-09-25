import argparse
import time
import torch

from Coloring.learning.utils.penv import ParallelEnv
from Coloring.learning.utils.storage import get_model_dir
from Coloring.learning.utils.env import make_env
from Coloring.learning.utils.other import seed, synthesize

from Coloring.learning.ppo.utils.agent import Agent

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--agents", default=1, type=int,
                    help="amount of agents")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
args = parser.parse_args()

# Set seed for all randomness sources

seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments
print("NAME:________________________  ", __name__)
if __name__ == '__main__':
    envs = []
    for i in range(args.procs):
        env = make_env(
            args.env, args.agents, seed=args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = get_model_dir(args.model)
    agents = []
    for agent in range(args.agents):
        agents.append(Agent(agent, env.observation_space, env.action_space, model_dir,
                            device=device))
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros((args.procs, args.agents), device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        joint_actions = []
        for agent_index, agent in enumerate(agents):
            actions = agent.get_actions(obss, agent_index)
            joint_actions.append(actions)

        obss, rewards, dones, _ = env.step(joint_actions)

        log_episode_return += torch.tensor(rewards,
                                           device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(
                    log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return_transposed = log_episode_return.transpose(
            0, 1) * mask
        log_episode_return = log_episode_return_transposed.transpose(0, 1)
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = synthesize(
        logs["return_per_episode"])
    num_frames_per_episode = synthesize(
        logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    # Print worst episodes

    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(
            len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i,
                  logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
