import argparse
import time
import datetime
import torch
import learning
# cd into storage and call either
# tensorboard --logdir ./ --host localhost --port 8888
# or
# python -m tensorboard.main --logdir ./ --host localhost --port 8888
import tensorboardX
import sys

import learning.utils
from learning.model import ACModel

# to show large tensors without truncation uncomment the next line
# torch.set_printoptions(threshold=10_000)

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--agents", default=1, type=int,
                    help="amount of agents")
parser.add_argument("--env", default='Empty-Grid-v0',
                    help="name of the environment to train on (default: empty grid size 5x5)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
# epochs range(3,30), wie oft anhand der experience gelernt wird?
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
# batch range(4, 4096) -> 256 insgesamt erhält man frames-per-proc*procs (128*16=2048) batch elemente und davon erhält man
# 2048/256 = 8 mini batches
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
# a Number that defines how often a (random) action is chosen for the batch/experience
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
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

args = parser.parse_args()
agents = args.agents
args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = learning.utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = learning.utils.get_txt_logger(model_dir)
csv_file, csv_logger = learning.utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

learning.utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    envs.append(learning.utils.make_env(
        args.env, args.agents, args.seed + 10000 * i))
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
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model
models = []
for agent in range(agents):
    model = ACModel(obs_space, envs[0].action_space)
    if "model_state" in status:
        model.load_state_dict(status["model_state"][agent])
    model.to(device)
    models.append(model)

txt_logger.info("Model loaded\n")
# txt_logger.info("{}\n".format(acmodel))

# Load algo
print("NAME:________________________  ", __name__)
if __name__ == '__main__':

    if args.algo == "a2c":
        algo = learning.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = learning.PPOAlgo(envs, models, device, args.frames_per_proc,
                                args.discount, args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
                                args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.epochs,
                                args.batch_size, preprocess_obss, None, args.agents)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:  # TODO
        for agent in range(agents):
            algo.optimizers[agent].load_state_dict(
                status["optimizer_state"][agent])

    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    # while num_frames >= args.frames:
    #     args.frames = args.frames*2  # add more frames

    while num_frames < args.frames:
        # Update model parameters

        # Log some values

        logs2 = {
            "entropy": [],
            "value": [],
            "policy_loss": [],
            "value_loss": [],
            "grad_norm": []
        }

        update_start_time = time.time()
        algo.prepare_experiences()
        for agent in range(agents):

            exps, logs1 = algo.collect_experience(agent)
            logs2 = algo.update_parameters(exps, agent, logs2)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = learning.utils.synthesize(
                logs["return_per_episode"])
            rreturn_per_episode = learning.utils.synthesize(
                logs["reshaped_return_per_episode"])
            num_frames_per_episode = learning.utils.synthesize(
                logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["reshaped_return_per_episode_" +
                       key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" +
                       key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()

            header += ["entropy_of_agent_" + str(agent)
                       for agent in range(agents)]
            data += [logs["entropy"][agent] for agent in range(agents)]
            header += ["value_of_agent_" + str(agent)
                       for agent in range(agents)]
            data += [logs["value"][agent] for agent in range(agents)]
            header += ["policy_loss_of_agent_" +
                       str(agent) for agent in range(agents)]
            data += [logs["policy_loss"][agent] for agent in range(agents)]
            header += ["grad_norm_of_agent_" +
                       str(agent) for agent in range(agents)]
            data += [logs["grad_norm"][agent] for agent in range(agents)]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | Return/Episode:σ μ min Max {:.2f} {:.2f} {:.2f} {:.2f} | numFrames:μ σ m M {:.1f} {:.1f} {} {}"
                .format(*data))

            txt_logger.info(str(("entropy per agent: ", logs["entropy"])))
            txt_logger.info(str(("value per agent: ", logs["value"])))
            txt_logger.info(
                str(("value loss per agent: ", logs["value_loss"])))
            txt_logger.info(str(("grad norm per agent: ", logs["grad_norm"])))

            header += ["return_per_episode_" +
                       key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": [models[agent].state_dict() for agent in range(agents)],
                      "optimizer_state": [algo.optimizers[agent].state_dict() for agent in range(agents)]}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            learning.utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
