import os
import argparse
from src.tools import Config
from src.trainer import Trainer
import json
from datetime import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_log', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:4')
    # parser.add_argument('--path_to_save_checkpoint', type=str, required=True)
    args = parser.parse_args()
    print(args.device)
    config_path = os.path.join(args.path_to_log, 'config.json')
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)
        
    config = Config(**config_dict)
    
    trainer = Trainer(config, device=args.device)

    trainer.test(args.path_to_log)