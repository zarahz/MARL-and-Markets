import argparse


def get_train_args(parser):
    '''
    Add PPO relevant training arguments to the parser.
    '''

    # epochs range(3,30), wie oft anhand der experience gelernt wird
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")

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
    # epsilon of clipping range(0.1,0.3)
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")

    return parser
