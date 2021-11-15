
def get_train_args(parser):
    '''
    Add DQN relevant training arguments to the parser.
    '''

    # epsilon for action selection
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="[DQN] Starting value of epsilon, used for action selection. (default: 1.0 -> high exploration)")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="[DQN] Ending value of epsilon, used for action selection. (default: 0.01 -> high exploitation)")
    parser.add_argument("--epsilon-decay", type=int, default=5000,
                        help="[DQN] Controls the rate of the epsilon decay in order to shift from exploration to exploitation. The higher the value the slower epsilon decays. (default: 5.000)")

    # memory settings
    parser.add_argument("--replay-size", type=int, default=40000,
                        help="[DQN] Size of the replay memory. (default: 40.000)")
    parser.add_argument("--initial-target-update", type=int, default=1000,
                        help="[DQN] Frames until the target network is updated, Needs to be smaller than --target-update! (default: 1.000)")
    parser.add_argument("--target-update", type=int, default=15000,
                        help="[DQN] Frames between updating the target network, Needs to be smaller or equal to --frames-per-proc and bigger than --initial-target-update! (default: 15.000)")

    return parser
