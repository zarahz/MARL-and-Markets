import argparse

from learning.utils.arguments import base_args


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = base_args(parser)

    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--capture-interval", type=int, default=10,
                        help="number of gif caputures of episodes (default: 10, 0 means no capturing)")
    parser.add_argument("--capture-frames", type=int, default=50,
                        help="number of frames in caputure (default: 50, 0 means no capturing)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=1000000,
                        help="number of frames of training (default: 1.000.000)")

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
    parser.add_argument("--frames-per-proc", type=int, default=1024,
                        help="number of frames per process before update (default: 1024)")
    # gamma = discount range(0.88,0.99) most common is 0.99
    parser.add_argument("--gamma", type=float, default=0.99,
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

    return args


def get_vis_args():
    parser = argparse.ArgumentParser()
    parser = base_args(parser)

    parser.add_argument("--shift", type=int, default=0,
                        help="number of times the environment is reset at the beginning (default: 0)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="select the action with highest probability (default: False)")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="pause duration between two consequent actions of the agent (default: 0.1)")
    parser.add_argument("--gif", type=str, default=None,
                        help="store output as gif with the given filename")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of episodes to visualize")

    args = parser.parse_args()
    return args
