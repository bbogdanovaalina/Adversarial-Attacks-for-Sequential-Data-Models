import os
import argparse
from src.tools import Config
from src.trainer import Trainer
import json
from src.tools import build_discr



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_log', type=str, required=True)
    parser.add_argument('--path_to_discr_log', type=str, required=True)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--attack', type=str, default='ifgsm')
    args = parser.parse_args()
    path_to_log = os.path.join(args.path_to_log, 'config.json')
    path_to_discr_log = os.path.join(args.path_to_discr_log, 'config.json')
    with open(path_to_log, 'r') as config_file:
        config_log_dict = json.load(config_file)

    with open(path_to_discr_log, 'r') as config_file:
        config_dicr_dict = json.load(config_file)

    config_log = Config(**config_log_dict)
    config_discr = Config(**config_dicr_dict)
    dicr = build_discr(config_discr, args.path_to_discr_log)
    trainer = Trainer(config_log)

    trainer.test_discr(args.path_to_log, dicr, epsilon=args.eps, max_iter=args.max_iter, attack=args.attack)