def base_args(parser):
    '''
    Basic arguments that are always needed for the environment and training/visualization
    '''
    # General parameters
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--agents", default=1, type=int,
                        help="amount of agents")
    parser.add_argument("--model", required=True,
                        help="name of the (trained) model")

    parser.add_argument("--capture", type=bool, default=True,
                        help="Boolean to enable capturing of environment and save as gif (default: True)")

    # Environment settings
    parser.add_argument("--env", default='Empty-Grid-v0',
                        help="name of the environment to train on (default: empty grid)")
    parser.add_argument("--agent-view-size", default=5, type=int,
                        help="grid size the agent can see, while standing in the middle (default: 5, so agent sees the 5x5 grid around him)")
    parser.add_argument("--grid-size", default=5, type=int,
                        help="size of the playing area (default: 5)")
    parser.add_argument("--max-steps", default=None, type=int,
                        help="max steps in environment to reach a goal")
    parser.add_argument("--setting", default="",
                        help="If set to mixed-motive the reward is not shared which enables a competitive environment (one vs. all). Another setting is percentage-reward, where the reward is shared (coop) and is based on the percanted of the grid coloration. The last option is mixed-motive-competitive which extends the normal mixed-motive setting by removing the field reset option. When agents run over already colored fields the field immidiatly change the color the one of the agent instead of resetting the color. (default: empty string - coop reward of one if the whole grid is colored)")
    parser.add_argument("--market", default='',
                        help="There are three options 'sm', 'am' and '' for none. SM = Shareholder Market where agents can auction actions similar to stocks. AM = Action Market where agents can buy specific actions from others. (Default = '')")
    parser.add_argument("--trading-fee", default=0.05, type=float,
                        help="If a trade is executed, this value determens the price (market type am) / share (market type sm) the agents exchange (Default: 0.05)")

    return parser
