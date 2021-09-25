import argparse

from Coloring.learning.utils.arguments import base_args, base_training_args


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = base_args(parser)
    parser = base_training_args(parser)

    # epsilon for action selection
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="starting value of epsilon, used for action selection (default: 0.9 -> high exploration)")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="ending value of epsilon, used for action selection (default: 0.05 -> high exploitation)")
    parser.add_argument("--epsilon-decay", type=int, default=10000,
                        help="Controls the rate of the epsilon decay in order to shift from exploration to exploitation. The higher the value the slower epsilon decays. (default: 1000)")

    # memory settings
    parser.add_argument("--initial-target-update", type=int, default=10000,
                        help="Frames until the target network is updated, Needs to be smaller than target update! (default: 10000)")
    parser.add_argument("--target-update", type=int, default=10*10000,
                        help="Frames between updating the target network, Needs to be smaller or equal to frames-per-proc and bigger than initial target update! (default: 100000 - 10 times the initial memory!)")

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
