import datetime
import math
import numpy as np

import tensorboardX
import torch

# general imports
from learning.utils.env import make_env
from learning.utils.format import get_obss_preprocessor
from learning.utils.other import seed
from learning.utils.storage import *

# ppo
from learning.ppo.algorithm import PPO
from learning.ppo.model import ACModel

# dqn
from learning.dqn.algorithm import DQN
from learning.dqn.model import DQNModel
from learning.dqn.utils.replay import ReplayMemory

from learning.utils.arguments import training_args

args = training_args()
agents = args.agents

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = get_txt_logger(model_dir)
csv_file, csv_logger = get_csv_logger(model_dir, "log")
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

if args.market:
    action_space = envs[0].action_space.nvec.prod()
else:
    action_space = envs[0].action_space.n

# Load models
ppo_models = []
dqn_policy_nets = []
dqn_target_nets = []

for agent in range(agents):
    if args.algo == "ppo":
        model = ACModel(obs_space, action_space)
        if "model_state" in status:
            model.load_state_dict(status["model_state"][agent])
        ppo_models.append(model)
    elif args.algo == "dqn":
        policy_net = DQNModel(obs_space, action_space)
        target_net = DQNModel(obs_space, action_space)
        if "policy_state" in status:
            policy_net.load_state_dict(status["policy_state"][agent])
        if "target_state" in status:
            target_net.load_state_dict(status["target_state"][agent])
        target_net.load_state_dict(policy_net.state_dict())
        dqn_policy_nets.append(policy_net)
        dqn_target_nets.append(target_net)

txt_logger.info("Models loaded\n")

# Load dqn
print("NAME:________________________  ", __name__)
if __name__ == '__main__':
    if args.algo == "ppo":
        ppo = PPO(envs, agents, ppo_models, device, args.frames_per_proc,
                  args.gamma, args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
                  args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.epochs,
                  args.batch_size, preprocess_obss)
        if "optimizer_state" in status:
            for agent in range(agents):
                ppo.optimizers[agent].load_state_dict(
                    status["optimizer_state"][agent])

    elif args.algo == "dqn":
        memory = ReplayMemory(args.replay_size)
        dqn = DQN(envs, agents, memory, dqn_policy_nets, dqn_target_nets,
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
                  initial_target_update=args.initial_target_update,
                  preprocess_obss=preprocess_obss,
                  action_space=action_space)

        if "optimizer_state" in status:
            for agent in range(agents):
                dqn.optimizers[agent].load_state_dict(
                    status["optimizer_state"][agent])

    txt_logger.info("Optimizer loaded\n")

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:

        # execute training and get logs
        if args.algo == "ppo":
            ppo_logs = {
                "entropy": [],
                "value": [],
                "policy_loss": [],
                "value_loss": [],
                "grad_norm": []
            }

            logs = ppo.run_and_log_parallel_envs(args.capture_frames)
            for agent in range(agents):
                exps = ppo.fill_and_reshape_experiences(agent)
                ppo_logs = ppo.optimize_model(exps, agent, ppo_logs)
            logs.update(ppo_logs)

        elif args.algo == "dqn":
            logs = dqn.run_and_log_parallel_envs(args.capture_frames)

            # log epsilon
            updated_frames = num_frames + logs["num_frames"]
            eps_threshold = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                math.exp(-1. * updated_frames / args.epsilon_decay)
            txt_logger.info("(EPSILON: " + str(eps_threshold) + ")")

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
            if args.algo == "ppo":
                status = {"num_frames": num_frames, "update": update,
                          "model_state": [ppo_models[agent].state_dict() for agent in range(agents)],
                          "optimizer_state": [ppo.optimizers[agent].state_dict() for agent in range(agents)]}
            elif args.algo == "dqn":
                status = {"num_frames": num_frames, "update": update,
                          "policy_state": [dqn_policy_nets[agent].state_dict() for agent in range(agents)],
                          "target_state": [dqn_target_nets[agent].state_dict() for agent in range(agents)],
                          "optimizer_state": [dqn.optimizers[agent].state_dict() for agent in range(agents)]}

            save_status(status, model_dir)
            txt_logger.info("Status saved")

        if args.capture_interval > 0 and update % args.capture_interval == 0 or num_frames > args.frames:
            # ensure saving of last round
            gif_name = str(update) + "_" + str(num_frames) + ".gif"
            save_capture(model_dir, gif_name, np.array(logs["capture_frames"]))

    # calculate how many episodes are max and min possible for the optimal and worst steps
    walkable_cells = len(envs[0].env.walkable_cells())
    best_case_steps = math.ceil(walkable_cells/args.agents)

    max_episodes = (
        int(args.frames_per_proc / best_case_steps) * args.procs) * args.log_interval
    min_episodes = (
        int(args.frames_per_proc / envs[0].env.max_steps) * args.procs) * args.log_interval
    txt_logger.info(
        f"Best case step use - count of episodes (per update): {max_episodes} \n")
    txt_logger.info(
        f"Worst case step use - count of episodes (per update): {min_episodes} \n")

    os._exit(1)
