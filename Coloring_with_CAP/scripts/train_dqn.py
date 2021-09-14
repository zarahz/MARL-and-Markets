import datetime
import math
import random
from itertools import count

import tensorboardX
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# from learning.dqn.config import *
# Training
from learning.dqn.utils import *
from learning.dqn.utils.arguments import get_train_args

from learning.dqn.utils.replay import ReplayMemory
from learning.dqn.model import DQN
from learning.utils.format import get_obss_preprocessor
from learning.utils.other import seed

from learning.utils.storage import *

args = get_train_args()
agents = args.agents

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_ppo_seed{args.seed}_{date}"
model_name = args.model or default_model_name
model_dir = get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = get_txt_logger(model_dir)
csv_file, csv_logger = get_csv_logger(model_dir, "log")
csv_rewards_file, csv_rewards_logger = get_csv_logger(
    model_dir, "rewards")
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))
# Set seed for all randomness sources

seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

env = make_env(
    args.env, args.agents,
    grid_size=args.grid_size,
    agent_view_size=args.agent_view_size,
    max_steps=args.max_steps,
    setting=args.setting,
    market=args.market,
    trading_fee=args.trading_fee,
    seed=args.seed + 10000)
txt_logger.info("Environment loaded\n")

# Load training status

try:
    status = get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "episode": 0, "update": 0, "csv_update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = get_obss_preprocessor(
    env.observation_space)
if args.market:
    action_space = env.action_space.nvec.prod()
else:
    action_space = env.action_space.n

txt_logger.info("Observations preprocessor loaded")

# Load model
policy_nets = []
target_nets = []
optimizers = []
for agent in range(agents):
    policy_net = DQN(
        obs_space, action_space).to(device)
    target_net = DQN(
        obs_space, action_space).to(device)
    if "target_state" in status:
        target_net.load_state_dict(status["target_state"][agent])
    if "policy_state" in status:
        policy_net.load_state_dict(status["policy_state"][agent])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())

    policy_nets.append(policy_net)
    target_nets.append(target_net)
    optimizers.append(optimizer)

memory = ReplayMemory(10000)

steps = 0


def select_action(state):
    """
    select_action will select an action accordingly to an epsilon greedy policy.
    Simply put, we’ll sometimes use our networks for choosing the action, and sometimes we’ll just sample one uniformly.
    The probability of choosing a random action will start at EPS_START
    and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
    """
    global steps
    sample = random.random()
    eps_threshold = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
        math.exp(-1. * steps / args.epsilon_decay)
    steps += 1
    if sample > eps_threshold:
        with torch.no_grad():
            joint_actions = []
            for agent in range(agents):
                obs = torch.tensor(state[agent]["image"], device=device, dtype=torch.float).transpose(
                    2, 1).transpose(0, 1)
                result = policy_net(obs.unsqueeze(0))
                joint_actions.append([result.max(1)[1].item()])  # .view(1, 1)
            return torch.tensor(joint_actions)
    else:
        return torch.tensor([[random.randrange(action_space) for _ in range(agents)]], device=device, dtype=torch.long)


def optimize_model():
    """
    Here, you can find an optimize_model function that performs a single step of the optimization. It first samples a
    batch, concatenates all the tensors into a single one, computes Q(st,at) and V(st+1)=maxaQ(st+1,a), and combines
    them into our loss. By defition we set V(s)=0 if s is a terminal state. We also use a target networks to compute
    V(st+1) for added stability. The target networks has its weights kept frozen most of the time, but is updated with
    the policy networks’s weights every so often. This is usually a set number of steps but we shall use episodes for
    simplicity.
    """
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.tensor([s[0]["image"] for s in batch.next_state
                                          if s is not None], device=device, dtype=torch.float)
    non_final_next_states = non_final_next_states.transpose(
        3, 2).transpose(1, 2)
    state_batch = torch.tensor([state[0]["image"]
                                for state in batch.state if state is not None], device=device, dtype=torch.float)
    # reshape into (batch_size, 3, 7 (agent_view), 7 (agent_view))
    state_batch = state_batch.transpose(3, 2).transpose(1, 2)
    # state_batch = torch.cat(state_batch)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # print("-------------------")
    # print("state_action_values:")
    # Compute Q(s_t, a) - the networks computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print(torch.mean(state_action_values))

    # print("next_state_values:")
    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # print(torch.mean(next_state_values))

    # print("expected_state_action_values:")
    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * args.gamma) + reward_batch
    # print(torch.mean(expected_state_action_values))

    # print("loss:")
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))
    # print(loss)
    # loss = F.smooth_l1_loss(state_action_values,
    #                         expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Main Training Loop
if __name__ == '__main__':
    num_episodes = 50
    update = status["update"]
    csv_update = status["csv_update"] if "csv_update" in status else 0
    start_time = time.time()

    # log values
    log_return = []

    log_fully_colored = 0
    log_coloration_percentage = []
    # variable to sum up all executed trades
    log_trades = torch.zeros(args.frames_per_proc, dtype=torch.int)
    # variable to save the trades for each episode
    log_episode_trades = torch.zeros(args.frames_per_proc, dtype=torch.int)
    # variable to sum up all reset fields
    log_num_reset_fields = torch.zeros(args.frames_per_proc, dtype=torch.int)
    # variable to save the reset fields for each episode
    log_episode_reset_fields = torch.zeros(
        args.frames_per_proc, dtype=torch.int)

    steps = status["num_frames"]
    while steps < args.frames:
        for episode in range(args.frames):

            # Initialize the environment and state
            state = env.reset()
            for t in count():
                # env.render('human')
                i = steps % args.frames_per_proc
                # Select and perform an action
                joint_actions = select_action(state)

                # print("step: ", env.env.step_count, ", action: ", joint_actions)

                obs, env_reward, done, info = env.step(joint_actions)

                # Update log values
                log_episode_reset_fields[i] = torch.tensor(
                    [info['reset_fields']], device=device)
                if 'trades' in info:
                    log_episode_trades[i] = torch.tensor(
                        [info['trades']], device=device)

                reward = torch.tensor([env_reward], device=device)

                # Observe new state
                next_state = None if done else obs

                # Store the transition in memory
                memory.push(state, joint_actions, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target networks)
                optimize_model()
                if done:
                    log_return.append(env_reward)
                    log_coloration_percentage.append(
                        info['coloration_percentage'])
                    log_fully_colored += info["fully_colored"]
                    # break

                 # Update the target networks
                # TARGET_UPDATE = giving your network more time to consider many
                # actions that have taken place recently instead of updating all the time
                if episode % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    update += 1

                if steps % args.save_interval == 0:
                    csv_update += 1
                    keep = int(args.frames_per_proc /
                               env.env.max_steps)*args.procs
                    # just to have similar statistis with ppo data!
                    for agent in range(agents):
                        agent_logs = {"reward_agent_"+str(agent): [episode_log_return[agent] for episode_log_return in log_return[-keep:]]
                                      }
                    logs = {
                        "num_frames": steps,
                        "trades": log_episode_trades[-keep:].tolist(),
                        "num_reset_fields": log_episode_trades[-keep:].tolist(),
                        "grid_coloration_percentage": log_coloration_percentage,
                        "fully_colored": log_fully_colored
                    }
                    logs.update(agent_logs)
                    header, data, reward_data = prepare_csv_data(
                        agents, logs, csv_update, steps, start_time=start_time, txt_logger=txt_logger)

                    update_csv_file(csv_file, csv_logger, csv_update,
                                    dict(zip(header, data)))
                    # reset everything
                    log_return = []

                    log_fully_colored = 0
                    log_coloration_percentage = []
                    # variable to sum up all executed trades
                    log_trades = torch.zeros(
                        args.frames_per_proc, dtype=torch.int)
                    # variable to save the trades for each episode
                    log_episode_trades = torch.zeros(
                        args.frames_per_proc, dtype=torch.int)
                    # variable to sum up all reset fields
                    log_num_reset_fields = torch.zeros(
                        args.frames_per_proc, dtype=torch.int)
                    # variable to save the reset fields for each episode
                    log_episode_reset_fields = torch.zeros(
                        args.frames_per_proc, dtype=torch.int)

                    if csv_update % 10 == 0:
                        status = {"num_frames": steps, "episode": episode, "update": update,
                                  "policy_state": [policy_nets[agent].state_dict() for agent in range(agents)],
                                  "target_state": [target_nets[agent].state_dict() for agent in range(agents)],
                                  "optimizer_state": [optimizer.state_dict()]}
                        save_status(status, model_dir)
                        txt_logger.info("Status saved")

                if done:
                    break

    print('Complete')
    # plt.ioff()
    # plt.show()
