
def get_train_args(parser):
    '''
    Add DQN relevant training arguments to the parser.
    '''

    # epsilon for action selection
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="starting value of epsilon, used for action selection (default: 0.9 -> high exploration)")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="ending value of epsilon, used for action selection (default: 0.05 -> high exploitation)")
    parser.add_argument("--epsilon-decay", type=int, default=5000,
                        help="Controls the rate of the epsilon decay in order to shift from exploration to exploitation. The higher the value the slower epsilon decays. (default: 5000)")

    # memory settings
    parser.add_argument("--replay-size", type=int, default=40000,
                        help="Size of the replay memory (default: 40.000)")
    parser.add_argument("--initial-target-update", type=int, default=1000,
                        help="Frames until the target network is updated, Needs to be smaller than target update! (default: 1000)")
    parser.add_argument("--target-update", type=int, default=15000,
                        help="Frames between updating the target network, Needs to be smaller or equal to frames-per-proc and bigger than initial target update! (default: 15000)")

    return parser
