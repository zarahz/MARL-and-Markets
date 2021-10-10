import argparse

from learning import dqn, ppo


def base_args():
    '''
    Basic arguments that are always needed for the environment and training/visualization
    '''
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--agents", default=1, type=int,
                        help="amount of agents")
    parser.add_argument("--model", default=None,
                        help="Name of the (trained) model, if none is given then a name is generated. (default: None)")

    parser.add_argument("--capture", type=bool, default=True,
                        help="Boolean to enable capturing of environment and save as gif (default: True)")

    # Environment settings
    parser.add_argument("--env", default='Empty-Grid-v0',
                        help="name of the environment to train on (default: empty grid)")
    parser.add_argument("--agent-view-size", default=7, type=int,
                        help="grid size the agent can see, while standing in the middle (default: 5, so agent sees the 5x5 grid around him)")
    parser.add_argument("--grid-size", default=9, type=int,
                        help="size of the playing area (default: 9)")
    parser.add_argument("--max-steps", default=None, type=int,
                        help="max steps in environment to reach a goal")
    parser.add_argument("--setting", default="",
                        help="If set to mixed-motive the reward is not shared which enables a competitive environment (one vs. all). Another setting is percentage-reward, where the reward is shared (coop) and is based on the percanted of the grid coloration. The last option is mixed-motive-competitive which extends the normal mixed-motive setting by removing the field reset option. When agents run over already colored fields the field immidiatly change the color the one of the agent instead of resetting the color. (default: empty string - coop reward of one if the whole grid is colored)")
    parser.add_argument("--market", default='',
                        help="There are three options 'sm', 'am' and '' for none. SM = Shareholder Market where agents can auction actions similar to stocks. AM = Action Market where agents can buy specific actions from others. (Default = '')")
    parser.add_argument("--trading-fee", default=0.1, type=float,
                        help="If a trade is executed, this value determens the price (market type am) / share (market type sm) the agents exchange (Default: 0.1)")

    return parser


def training_args():
    '''
    Define all training arguments that are currently needed to configure the two algorithms (ppo & dqn)
    '''

    # --------------------------------------------------
    # Basic arguments needed by both learning algorithms (environment settings etc.)
    # --------------------------------------------------
    parser = base_args()

    # --------------------------------------------------
    # arguments needed by both learning algorithms
    # --------------------------------------------------
    # TODO Move to base_args to be used for visualization as well!
    parser.add_argument("--algo", required=True,
                        help="Algorithm to use for training. Choose between 'ppo' and 'dqn'.")

    parser.add_argument("--frames", type=int, default=1000000,
                        help="number of frames of training (default: 1.000.000)")
    # i.e. frames-per-proc = 128 that means 128 times the (--procs=16) parallel envs are played through and logged.
    # If max_steps = 25 the environment can at least finish 5 times (done if max step is reached)
    # and save its rewards, that means there are at least 5*16=80 rewards
    parser.add_argument("--frames-per-proc", type=int, default=1024,
                        help="number of frames per process before update (default: 1024)")
    parser.add_argument("--procs", type=int, default=16,
                        help="Number of processes/environments running parallel (default: 16)")

    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")

    # i.e. batch_size = 256: overall one run contains frames-per-proc*procs (128*16=2048) batch elements / Transitions
    # and out of that 2048/256 = 8 mini batches can be drawn
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for dqn (default: ppo 256, dqn 128)")

    # gamma = discount range(0.88,0.99) most common is 0.99
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor (default: 0.99)")

    # intervals
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of frames between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--capture-interval", type=int, default=10,
                        help="number of gif caputures of episodes (default: 10, 0 means no capturing)")
    parser.add_argument("--capture-frames", type=int, default=50,
                        help="number of frames in caputure (default: 50, 0 means no capturing)")

    # optimizer values
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")

    # --------------------------------------------------
    # Expand parser with specific arguments for each algorithm
    # --------------------------------------------------
    parser = ppo.utils.arguments.get_train_args(parser)

    parser = dqn.utils.arguments.get_train_args(parser)

    args = parser.parse_args()

    return args


def vis_args():
    '''
    Define all visualization arguments that are currently needed to configure the two algorithms (ppo & dqn)
    '''

    # --------------------------------------------------
    # Basic arguments needed by both learning algorithms (environment settings etc.)
    # --------------------------------------------------
    parser = base_args()

    # --------------------------------------------------
    # arguments needed by both learning algorithms
    # --------------------------------------------------
    # TODO Move to base_args to be used for visualization as well!
    # parser.add_argument("--algo", required=True,
    #                     help="Algorithm to use for training. Choose between 'ppo' and 'dqn'.")

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

    # --------------------------------------------------
    # Expand parser with specific arguments for each algorithm
    # --------------------------------------------------
    # parser = ppo.utils.arguments.get_train_args(parser)

    # parser = dqn.utils.arguments.get_train_args(parser)

    args = parser.parse_args()

    return args
