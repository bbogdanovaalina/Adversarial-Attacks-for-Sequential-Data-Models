import os
import argparse
from src.tools import Config
from src.trainer import Trainer
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_log', type=str, required=True)
    parser.add_argument('--eps', type=int, default=0.5)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--attack', type=str, default='ifgsm')
    args = parser.parse_args()

    config_path = os.path.join(args.path_to_log, 'config.json')
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)
        
    config = Config(**config_dict)
    
    trainer = Trainer(config)

    trainer.test_adverasial(args.path_to_log, epsilon=args.eps, max_iter=args.max_iter, attack=args.attack)