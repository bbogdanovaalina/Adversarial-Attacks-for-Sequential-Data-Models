import os
import argparse
from src.tools import Config
from src.trainer import Trainer
import json
from datetime import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--path_to_save_checkpoint', type=str, default='./logs')
    parser.add_argument('--add_info', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:5')
    
    args = parser.parse_args()
    print(args.device)
    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
        
    config = Config(**config_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(args.path_to_save_checkpoint):
        os.mkdir(args.path_to_save_checkpoint)
    
    if args.add_info is not None:


        path_to_save = f'{args.path_to_save_checkpoint}/{args.add_info}_{config.model.model_name}_{config.dataset_name}_{timestamp}'

    else:
        path_to_save = f'{args.path_to_save_checkpoint}/{config.model.model_name}_{config.dataset_name}_{timestamp}'

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    with open(os.path.join(path_to_save, 'config.json') , 'w') as json_file:
        json.dump(config_dict, json_file)

    
    trainer = Trainer(config, device=args.device)

    trainer.train(path_to_save)

