import os
import argparse
from src.tools import Config, build_discr
from src.trainer import Trainer
import json
import torch
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_log', type=str, required=True)
    parser.add_argument('--eps', type=float, default=0.9)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--lamb', type=float, default=None)
    parser.add_argument('--attack', type=str, default='deepfool')
    parser.add_argument('--discr_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--is_train', type=int, default=1)
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

    trainer.test_adverasial(args.path_to_log, epsilon=args.eps, 
                            max_iter=args.max_iter,
                            attack=args.attack, 
                            discr=discr, 
                            lamb=args.lamb, 
                            is_train=args.is_train)
    


    folder_path = args.path_to_log +"/data/"

    path_to_data = "TRAIN_"+f"{args.attack}_eps{args.eps}_mi{args.max_iter}"
    path_per = '/per_data_list.pth'
    path_true = '/initial_data_list.pth'
    per = torch.load(folder_path+path_to_data+path_per)
    true = torch.load(folder_path+path_to_data+path_true)

    per_pd = pd.DataFrame(per.squeeze(-1))
    per_class = torch.ones(per_pd.shape[0]).long()
    per_pd['class'] = per_class

    true_pd = pd.DataFrame(true.squeeze(-1))
    true_class = torch.zeros(true_pd.shape[0]).long()
    true_pd['class'] = true_class
    fin = pd.concat([per_pd, true_pd])
    fin.to_csv(folder_path+path_to_data+'/dataset_'+path_to_data+'.csv', index=False)
