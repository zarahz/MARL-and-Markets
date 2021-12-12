import argparse

from learning import dqn, ppo


def base_args():
    '''
    Basic arguments that are always needed for the environment and training/visualization
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", required=True,
                        help="Algorithm to use for training. Choose between 'ppo' and 'dqn'.")

    # General parameters
    parser.add_argument("--seed", type=int, default=1,
                        help="Generate the same set of pseudo random constellations, colors, positions, etc. every time the algorithm is executed. (default: 1)")
    parser.add_argument("--agents", default=2, type=int,
                        help="Amount of agents. (default: 2)")
    parser.add_argument("--model", default=None,
                        help="Path of the model inside the storage folder, if none is given then a random name is generated. (default: None)")

    parser.add_argument("--capture", type=bool, default=True,
                        help="Boolean to enable capturing of the environment. The outcome are in form of gifs. (default: True)")

    # Environment settings
    parser.add_argument("--env", default='Empty-Grid-v0',
                        help="Environment ID, choose between Empty-Grid-v0 for an empty environment and FourRooms-Grid-v0 for an environment divided into equal sized rooms. (default: Empty-Grid-v0)")
    parser.add_argument("--agent-view-size", default=7, type=int,
                        help="Grid size the agent can see. Agent Observation is based on that field of view. For example, 7x7 grid size means agent can see three tiles in each direction. (default: 7)")
    parser.add_argument("--grid-size", default=5, type=int,
                        help="Size of the environment grid. (default: 5)")
    parser.add_argument("--max-steps", default=None, type=int,
                        help="Maximum amount of steps an agent has to reach a goal. If none is given then this max count is set to: grid size * grid size. (default: None)")
    parser.add_argument("--setting", default="",
                        help="Setting can be either: '' for cooperation, 'mixed-motive' for a mixed motive environment, 'mixed-motive-competitive' for a competitive composition or 'difference-reward' for a setting that calculates difference rewards. Cooperation means all agents get the same reward. If set to mixed-motive or mixed-motive-competitve the reward is not shared and each agent is responsible for its own success. In competitive mode, agents can take over opponent coloration without resetting the cells, otherwise cells are always reset when colored and walked over. The last option 'difference-reward' is a cooperation setting but calculates the reward for each agent by subtracting a new reward from the total reward. The new reward just excludes the action of this one agent. A high difference reward means, that the action of that agent was good. (default: '' for cooperation)")
    parser.add_argument("--market", default='',
                        help="There are three options: 'sm', 'am' and '' for none. SM = Shareholder Market where agents can sell or buy shares on the market. AM = Action Market where agents can buy specific actions from others. (default = '')")
    parser.add_argument("--trading-fee", default=0.1, type=float,
                        help="If a market transaction is executed, this value determines the price, i.e. in an action market this defines the price the buyer pays. In a shareholder market this value defines the share value. (default: 0.1)")

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

    parser.add_argument("--frames", type=int, default=80000,
                        help="Number of frames of training. (default: 80.000)")
    # i.e. frames-per-proc = 128 that means 128 times the (--procs=16) parallel envs are played through and logged.
    # If max_steps = 25 the environment can at least finish 5 times (done if max step is reached)
    # and save its rewards, that means there are at least 5*16=80 rewards
    parser.add_argument("--frames-per-proc", type=int, default=128,
                        help="Number of frames per process. In case of PPO this is the number of steps, before the model is optimized. (default: 128)")
    parser.add_argument("--procs", type=int, default=16,
                        help="Number of processes/environments running parallel. (default: 16)")

    parser.add_argument("--recurrence", type=int, default=1,
                        help="Number of time-steps the gradient is back propagated. If it is greater than one, a LSTM is added to the model to have memory. (default: 1)")

    # i.e. batch_size = 256: overall one run contains frames-per-proc*procs (128*16=2048) batch elements / Transitions
    # and out of that 2048/256 = 8 mini batches can be drawn
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size that is used for sampling.(default: 64)")

    # gamma = discount range(0.88,0.99) most common is 0.99
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor with 0 <= gamma < 1, specify how important future estimated rewards are. High value means high importance. (default: 0.99)")

    # intervals
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Number of frames between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Number of times the --frames-per-proc amount of frames needs to be reached, to log the current training values, i.e. rewards, into a csv file. (default: 10, 0 means no saving)")
    parser.add_argument("--capture-interval", type=int, default=10,
                        help="Number of times --frames-per-proc amount of frames needs to be reached, to capture the last --capture-frames amount of steps into a gif. Warning: --capture needs to be set to True as well. (default: 10, 0 means no capturing)")
    parser.add_argument("--capture-frames", type=int, default=50,
                        help="Number of frames that are captured. (default: 50, 0 means no capturing)")

    # optimizer values
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate. (default: 0.001)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Epsilon value for the Adam optimizer. (default: 1e-8)")

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

    parser.add_argument("--shift", type=int, default=0,
                        help="number of times the environment is reset at the beginning (default: 0)")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="pause duration between two consequent actions of the agent (default: 0.1)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of episodes to visualize")

    # --------------------------------------------------
    # Expand parser with specific arguments for each algorithm
    # --------------------------------------------------
    # parser = ppo.utils.arguments.get_train_args(parser)

    # parser = dqn.utils.arguments.get_train_args(parser)

    args = parser.parse_args()

    return args