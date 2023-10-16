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
    parser.add_argument('--attack', type=str, default='ifgsm_discr')
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--discr_path', type=str, default=None)
    parser.add_argument('--lamb', type=float, default=None)

    args = parser.parse_args()
    print(args.device)

    path_to_discr_log = os.path.join(args.path_to_discr_log, 'config.json')
    with open(path_to_discr_log, 'r') as config_file:
        config_dicr_dict = json.load(config_file)

    config_discr = Config(**config_dicr_dict)
    discr_critic = build_discr(config_discr, args.path_to_discr_log, device=args.device)


    path_to_log = os.path.join(args.path_to_log, 'config.json')
    if args.discr_path is not None:
        discr = []
        with open(args.discr_path, 'r') as f:
            for i in f:
                config_path = os.path.join(i.strip(), 'config.json')
                with open(config_path, 'r') as config_file:
                    config_dict = json.load(config_file)

                config = Config(**config_dict)
                discr.append(build_discr(config, i.strip(), device=args.device))
    else:
        discr = None


    with open(path_to_log, 'r') as config_file:
        config_log_dict = json.load(config_file)

    config_log = Config(**config_log_dict)
    trainer = Trainer(config_log, device=args.device)
    trainer.test_discr(args.path_to_log, discr_critic, epsilon=args.eps, max_iter=args.max_iter, attack=args.attack, discrim_adv=discr, lamb=args.lamb)