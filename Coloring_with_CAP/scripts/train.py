import argparse
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

import learning.ppo.utils
import learning.utils
from learning.ppo.model import ACModel
from learning.ppo.utils.storage import prepare_csv_data, save_capture, update_csv_file

# to show large tensors without truncation uncomment the next line
# torch.set_printoptions(threshold=10_000)

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--agents", default=1, type=int,
                    help="amount of agents")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

# Environment settings
parser.add_argument("--env", default='Empty-Grid-v0',
                    help="name of the environment to train on (default: empty grid size 5x5)")
parser.add_argument("--grid-size", default=5, type=int,
                    help="size of the playing area (default: 5)")
parser.add_argument("--percentage-reward", default=False,
                    help="reward agents based on percentage of coloration in the grid (default: False)")
parser.add_argument("--mixed-motive", default=False,
                    help="If set to true the reward is not shared which enables a mixed motive environment (one vs. all). Otherwise agents need to work in cooperation to gain more reward. (default: False = Cooperation)")
parser.add_argument("--market", default='',
                    help="There are three options 'sm', 'am' and '' for none. SM = Shareholder Market where agents can auction actions similar to stocks. AM = Action Market where agents can buy specific actions from others. (Default = '')")
parser.add_argument("--trading-fee", default=0.05, type=float,
                    help="If a trade is executed, this value determens the price (market type am) / share (market type sm) the agents exchange (Default: 0.05)")


parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--capture-interval", type=int, default=10,
                    help="number of gif caputures of episodes (default: 10, 0 means no capturing)")
parser.add_argument("--capture-frames", type=int, default=50,
                    help="number of frames in caputure (default: 50, 0 means no capturing)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--capture", type=bool, default=True,
                    help="Boolean to enable capturing of environment and save as gif (default: True)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
# epochs range(3,30), wie oft anhand der experience gelernt wird?
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
# batch range(4, 4096) -> 256 insgesamt erhält man frames-per-proc*procs (128*16=2048) batch elemente / Transitions
# und davon erhält man 2048/256 = 8 mini batches
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
# a Number that defines how often a (random) action is chosen for the batch/experience
# i.e. frames-per-proc = 128 that means 128 times the (16) parallel envs are played through and logged (in func prepare_experiences).
# If max_steps = 25 the environment can at least finish 5 times (done if max step is reached)
# and save its rewards that means there are at least 5*16=80 rewards
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
# gamma = discount range(0.88,0.99) most common is 0.99
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
# GAE = Generalized advantage estimator wird in verbindung mit dem advantage estimator berechnet
# Â von GAE(delta, lambda) zum zeitpunkt t = Summe (lambda*gamma)^l * delta zum zeitpunkt (t+l) ^ V
# range(0.9,1)
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
# entropy coef -> c2 * S[pi von theta](state t)
# with S as "entropy Bonus"
# range(0, 0.01)
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
# value function coef -> c1 * Loss func von VF zum Zeitpunkt t
# with LVF in t = (Vtheta(state t) - Vt ^ targ)^2 => squared error loss
# range(0.5,1)
# nötig wenn parameter zwischen policy und value funct. geteilt werden
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
# epsilon of clipping range(0.1,0.3)
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
# neural net training fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")

args = parser.parse_args()
agents = args.agents

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = learning.ppo.utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = learning.ppo.utils.get_txt_logger(model_dir)
csv_file, csv_logger = learning.ppo.utils.get_csv_logger(model_dir, "log")
csv_rewards_file, csv_rewards_logger = learning.ppo.utils.get_csv_logger(
    model_dir, "rewards")
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

learning.ppo.utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    envs.append(learning.ppo.utils.make_env(
        args.env, args.agents,
        grid_size=args.grid_size,
        percentage_reward=args.percentage_reward,
        mixed_motive=args.mixed_motive,
        market=args.market,
        trading_fee=args.trading_fee,
        seed=args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = learning.ppo.utils.get_status(model_dir)
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


# Load algo
print("NAME:________________________  ", __name__)
if __name__ == '__main__':
    algo = learning.ppo.PPOAlgo(envs, agents, models, device, args.frames_per_proc,
                                args.discount, args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
                                args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.epochs,
                                args.batch_size, preprocess_obss)

    if "optimizer_state" in status:  # TODO
        for agent in range(agents):
            algo.optimizers[agent].load_state_dict(
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

        update_start_time = time.time()
        logs = algo.prepare_experiences(args.capture_frames)
        # logs = {}
        for agent in range(agents):
            exps, logs1 = algo.collect_experience(agent)
            logs.update(logs1)
            log_per_agent = algo.update_parameters(exps, agent, log_per_agent)
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
            update_csv_file(csv_rewards_file,
                            csv_rewards_logger, update, reward_data)

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": [models[agent].state_dict() for agent in range(agents)],
                      "optimizer_state": [algo.optimizers[agent].state_dict() for agent in range(agents)]}
            learning.ppo.utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
        if args.capture_interval > 0 and update % args.capture_interval == 0 or num_frames > args.frames:
            # ensure saving of last round
            gif_name = str(update) + "_" + str(num_frames) + ".gif"
            save_capture(model_dir, gif_name, np.array(logs["capture_frames"]))
