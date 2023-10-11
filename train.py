import os
import argparse
from src.tools import Config
from src.trainer import Trainer
import json
from datetime import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--path_to_save_checkpoint', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
        
    config = Config(**config_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    path_to_save = f'./logs/{config.model.model_name}_{config.dataset_name}_{timestamp}'

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    with open(os.path.join(path_to_save, 'config.json') , 'w') as json_file:
        json.dump(config_dict, json_file)

    
    trainer = Trainer(config)

    trainer.train(path_to_save)

