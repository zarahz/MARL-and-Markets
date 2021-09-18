
import os
import time
import datetime
import numpy as np
import torch
# import learning
# cd into storage and call either
# tensorboard --logdir ./ --host localhost --port 8888
# or
# python -m tensorboard.main --logdir ./ --host localhost --port 8888
import tensorboardX
import sys
from learning.ppo.algorithm import PPO

import learning.ppo.utils
from learning.ppo.utils.arguments import get_train_args
import learning.utils
from learning.ppo.model import ACModel
from learning.utils.other import seed
from learning.utils.storage import prepare_csv_data, save_capture, update_csv_file

# to show large tensors without truncation uncomment the next line
# torch.set_printoptions(threshold=10_000)

args = get_train_args()
agents = args.agents

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_ppo_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = learning.utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = learning.utils.get_txt_logger(model_dir)
csv_file, csv_logger = learning.utils.get_csv_logger(model_dir, "log")
csv_rewards_file, csv_rewards_logger = learning.utils.get_csv_logger(
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
    envs.append(learning.utils.make_env(
        args.env, args.agents,
        grid_size=args.grid_size,
        agent_view_size=args.agent_view_size,
        max_steps=args.max_steps,
        setting=args.setting,
        market=args.market,
        trading_fee=args.trading_fee,
        seed=args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = learning.utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = learning.utils.get_obss_preprocessor(
    envs[0].observation_space)
txt_logger.info("Observations preprocessor loaded")

# Load model
models = []
for agent in range(agents):
    if args.market:
        action_space = envs[0].action_space.nvec.prod()
    else:
        action_space = envs[0].action_space.n
    model = ACModel(obs_space, action_space)
    if "model_state" in status:
        model.load_state_dict(status["model_state"][agent])
    model.to(device)
    models.append(model)

txt_logger.info("Model loaded\n")


# Load ppo
print("NAME:________________________  ", __name__)
if __name__ == '__main__':
    ppo = PPO(envs, agents, models, device, args.frames_per_proc,
              args.gamma, args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
              args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.epochs,
              args.batch_size, preprocess_obss)

    if "optimizer_state" in status:  # TODO
        for agent in range(agents):
            ppo.optimizers[agent].load_state_dict(
                status["optimizer_state"][agent])

    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        # Log some values
        log_per_agent = {
            "entropy": [],
            "value": [],
            "policy_loss": [],
            "value_loss": [],
            "grad_norm": []
        }

        logs = ppo.prepare_experiences(args.capture_frames)
        # logs = {}
        for agent in range(agents):
            exps, logs1 = ppo.collect_experience(agent)
            logs.update(logs1)
            log_per_agent = ppo.update_parameters(exps, agent, log_per_agent)
        logs.update(log_per_agent)
        # update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            header, data, reward_data = prepare_csv_data(
                agents, logs, update, num_frames, start_time=start_time, txt_logger=txt_logger)

            update_csv_file(csv_file, csv_logger, update,
                            dict(zip(header, data)))
            # update_csv_file(csv_rewards_file,
            #                 csv_rewards_logger, update, reward_data)

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": [models[agent].state_dict() for agent in range(agents)],
                      "optimizer_state": [ppo.optimizers[agent].state_dict() for agent in range(agents)]}
            learning.utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        if args.capture_interval > 0 and update % args.capture_interval == 0 or num_frames > args.frames:
            # ensure saving of last round
            gif_name = str(update) + "_" + str(num_frames) + ".gif"
            save_capture(model_dir, gif_name, np.array(logs["capture_frames"]))

    os._exit(1)
