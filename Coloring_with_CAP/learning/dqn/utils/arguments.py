import argparse

from learning.utils.arguments import base_args


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = base_args(parser)

    parser.add_argument("--frames", type=int, default=1000000,
                        help="number of frames of training (default: 1000000)")
    # a Number that defines how often a (random) action is chosen for the batch/experience
    # i.e. frames-per-proc = 128 that means 128 times the (16) parallel envs are played through and logged (in func prepare_experiences).
    # If max_steps = 25 the environment can at least finish 5 times (done if max step is reached)
    # and save its rewards that means there are at least 5*16=80 rewards
    parser.add_argument("--frames-per-proc", type=int, default=1024,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--procs", type=int, default=16,
                        help="[ONLY needed for logging similar data with ppo] number of processes (default: 16)")
    parser.add_argument("--log-interval", type=int, default=16384,
                        help="number of frames between two logs (default: 16384)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--capture-interval", type=int, default=10,
                        help="number of gif caputures of episodes (default: 10, 0 means no capturing)")
    parser.add_argument("--capture-frames", type=int, default=50,
                        help="number of frames in caputure (default: 50, 0 means no capturing)")

    # gamma = discount range(0.88,0.99) most common is 0.99
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    # batch range(4, 4096) -> 256 insgesamt erhält man frames-per-proc*procs (128*16=2048) batch elemente / Transitions
    # und davon erhält man 2048/256 = 8 mini batches
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size for dqn (default: 128)")
    # epsilon for action selection
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="starting value of epsilon, used for action selection (default: 0.9 -> high exploration)")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="ending value of epsilon, used for action selection (default: 0.05 -> high exploitation)")
    parser.add_argument("--epsilon-decay", type=int, default=10000,
                        help="Controls the rate of the epsilon decay in order to shift from exploration to exploitation. The higher the value the slower epsilon decays. (default: 1000)")
    parser.add_argument("--target-update", type=int, default=1000,
                        help="Steps (?) between updating the target network (default: 1000)")

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
