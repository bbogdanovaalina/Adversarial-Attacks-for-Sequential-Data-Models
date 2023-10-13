import os
import argparse
from src.tools import Config, build_discr
from src.trainer import Trainer
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_log', type=str, required=True)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--max_iter', type=int, default=2)
    parser.add_argument('--lamb', type=int, default=0.001)
    parser.add_argument('--attack', type=str, default='ifgsm')
    parser.add_argument('--discr_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:4')
    args = parser.parse_args()
    print(args.device)
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

    config_path = os.path.join(args.path_to_log, 'config.json')
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)

    # print(discr)
        
    config = Config(**config_dict)
    
    trainer = Trainer(config, device=args.device)

    trainer.test_adverasial(args.path_to_log, epsilon=args.eps, max_iter=args.max_iter, attack=args.attack, discr=discr, lamb=args.lamb)