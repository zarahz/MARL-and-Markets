import datetime
import math
import random
from itertools import count
import numpy as np

import tensorboardX
import torch
import torch.nn.functional as F
from learning.dqn.algorithm import DQN

# from learning.dqn.config import *
# Training
from learning.dqn.utils import *
from learning.dqn.utils.arguments import get_train_args

from learning.dqn.utils.replay import ReplayMemory
from learning.dqn.model import DQNModel
from learning.utils.format import get_obss_preprocessor
from learning.utils.other import seed

from learning.utils.storage import *

# TODO fix bug, algorithm never stops!
# TODO increase memory buffer (agent cannot learn if it is very forgetfull)
# TODO implement multi agents


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

# Load environments

envs = []
for i in range(args.procs):
    envs.append(make_env(
        args.env, args.agents,
        grid_size=args.grid_size,
        agent_view_size=args.agent_view_size,
        max_steps=args.max_steps,
        setting=args.setting,
        market=args.market,
        trading_fee=args.trading_fee,
        seed=args.seed + 10000))
txt_logger.info("Environment loaded\n")

# Load training status

try:
    status = get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = get_obss_preprocessor(
    envs[0].observation_space)
txt_logger.info("Observations preprocessor loaded")

# Load model
policy_nets = []
target_nets = []
for agent in range(agents):
    if args.market:
        action_space = envs[0].action_space.nvec.prod()
    else:
        action_space = envs[0].action_space.n
    policy_net = DQNModel(obs_space, action_space)
    target_net = DQNModel(obs_space, action_space)
    if "policy_state" in status:
        policy_net.load_state_dict(status["policy_state"][agent])
    if "target_state" in status:
        target_net.load_state_dict(status["target_state"][agent])
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.to(device)
    target_net.to(device)
    policy_nets.append(policy_net)
    target_nets.append(target_net)

txt_logger.info("Models loaded\n")

memory = ReplayMemory(args.target_update, agents,
                      args.agent_view_size, args.procs, device)


# Load dqn
print("NAME:________________________  ", __name__)
if __name__ == '__main__':
    dqn = DQN(envs, agents, memory, policy_nets, target_nets,
              device=device,
              num_frames_per_proc=args.frames_per_proc,
              gamma=args.gamma,
              lr=args.lr,
              batch_size=args.batch_size,
              epsilon_start=args.epsilon_start,
              epsilon_end=args.epsilon_end,
              epsilon_decay=args.epsilon_decay,
              adam_eps=args.optim_eps,
              target_update=args.target_update,
              preprocess_obss=preprocess_obss,
              action_space=action_space)

    if "optimizer_state" in status:  # TODO
        for agent in range(agents):
            dqn.optimizers[agent].load_state_dict(
                status["optimizer_state"][agent])

    txt_logger.info("Optimizer loaded\n")

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:

        logs = dqn.train(args.capture_frames)
        num_frames += logs["num_frames"]
        update += 1

        if update % args.log_interval == 0:
            header, data, reward_data = prepare_csv_data(
                agents, logs, update, num_frames, start_time=start_time, txt_logger=txt_logger)

            update_csv_file(csv_file, csv_logger, update,
                            dict(zip(header, data)))

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "policy_state": [policy_nets[agent].state_dict() for agent in range(agents)],
                      "target_state": [target_nets[agent].state_dict() for agent in range(agents)],
                      "optimizer_state": [dqn.optimizers[agent].state_dict() for agent in range(agents)]}
            save_status(status, model_dir)
            txt_logger.info("Status saved")

        if args.capture_interval > 0 and update % args.capture_interval == 0 or num_frames > args.frames:
            # ensure saving of last round
            gif_name = str(update) + "_" + str(num_frames) + ".gif"
            save_capture(model_dir, gif_name, np.array(logs["capture_frames"]))

    os._exit(1)
