import torch
from .tools import EarlyStopping, set_lr, cosine_annealing_lr, metrics, print_dict, calc_accuracy, print_log, set_model_to_eval_mode
from .attacks import get_attack
from .datasets import get_dataset_loader
from .models import get_model
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = self._build_model(config.model).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adamax(self.model.parameters(), lr=config.lr)
        
    def _build_model(self, config):
        return get_model(config.model_name)(config)
    
    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        preds = self.model(batch_x)
        loss = self.criterion(preds, batch_y)
        preds_class = preds.argmax(1)
        acc = calc_accuracy(batch_y.cpu().detach(), preds_class.cpu().detach())
        return loss, acc, preds

    def train(self, path):
        self.model.train()
        train_set, train_loader = get_dataset_loader(self.config, 'TRAIN')
        
        val_set, val_loader = get_dataset_loader(self.config, 'TEST')
        
        early_stopping = EarlyStopping(self.config.patience, self.config.verbose, self.config.delta)
        
        num_epochs = self.config.num_epochs
        path_to_log = os.path.join(path, 'training.txt')
        for epoch in tqdm(range(num_epochs)):
            train_loss = []
            val_loss = []
            acc_val_epoch = 0
            acc_train_epoch= 0
            
            print_log(path_to_log, ">>>>>>>>>>Trainig<<<<<<<<<<\n")
            for batch_x, batch_y in train_loader:
                loss, acc_train, preds_train = self._process_one_batch(batch_x, batch_y)
                train_loss.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                acc_train_epoch += acc_train

            acc_train_epoch = acc_train_epoch/ len(train_loader)
            print_log(path_to_log, f'Epoch {epoch + 1}/{num_epochs}: train_metrics: {acc_train_epoch}, train_loss_epoch: {np.mean(train_loss):.4f}\n')
            
            print_log(path_to_log, ">>>>>>>>>>Validation<<<<<<<<<<\n")
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    loss, acc_val, preds_val = self._process_one_batch(batch_x, batch_y)
                    val_loss.append(loss.item())
                    acc_val_epoch += acc_val

                acc_val_epoch = acc_val_epoch / len(val_loader)
            
            print_log(path_to_log, f'Epoch {epoch + 1}/{num_epochs}: val metrics: {acc_val_epoch}, val_loss_epoch: {np.mean(val_loss):.4f}\n')
            early_stopping(-acc_val_epoch, self.model, path)
            if early_stopping.early_stop:
                print_log(path_to_log, f'Early stopping')
                break

            set_lr(self.optim, cosine_annealing_lr(epoch, num_epochs, self.config.lr), path=path_to_log)

        self.test(path)
        


    def test(self, path):
        test_set, test_loader = get_dataset_loader(self.config, 'TEST')
        test_loss = []
        path_to_log = os.path.join(path, 'testing.txt')
        print_log(path_to_log, 'Loading model....\n')
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))
        print_log(path_to_log,'Loaded\n')
        print_log(path_to_log, ">>>>>>>>>>Testing<<<<<<<<<<\n")
        preds_list = []
        true_list = []
        with torch.no_grad():
            self.model.train(False)
            for batch_x, batch_y in test_loader:
                loss, acc_test, preds_test = self._process_one_batch(batch_x, batch_y)
                test_loss.append(loss.item())
                preds_list.extend(preds_test.argmax(1).cpu().tolist())
                true_list.extend(batch_y.cpu().tolist())

            m = metrics(true_list, preds_list)
            test_loss = np.mean(test_loss)
        to_print = {
            'Loss_test': test_loss, **m
        }
        print_log(path_to_log, f'{print_dict(**to_print)}\n\n')


    def test_adverasial(self, path, epsilon, max_iter, attack, discr = None, lamb = None, is_train = 1):
        if is_train == 1:
            dataset, loader = get_dataset_loader(self.config, 'TRAIN')
            add_info = 'TRAIN'
        else:
            dataset, loader = get_dataset_loader(self.config, 'TEST')
            add_info = 'TEST'
        path_to_log = os.path.join(path, 'testing_adversarial.txt')
        print_log(path_to_log, 'Loading model....\n')
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))
        print_log(path_to_log,'Loaded')
        print_log(path_to_log, "\n>>>>>>>>>>Testing_adversarial<<<<<<<<<<\n")
        print_log(path_to_log, f"{attack} attack\n")
        preds = []
        trues = []
        per_data_list = []
        initial_examples = []
        attack_func = get_attack(attack)
        # self.model.eval()
        # set_model_to_eval_mode(self.model)
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if attack == 'ifgsm_discr':
                per_data = attack_func(self.model, discr,
                                input=batch_x, 
                                target= batch_y,  
                                epsilon=epsilon, 
                                criterion=self.criterion, 
                                max_iter=max_iter, lamb = lamb)
            else:
                per_data = attack_func(self.model, 
                                    input=batch_x, 
                                    target= batch_y,  
                                    epsilon=epsilon, 
                                    criterion=self.criterion, 
                                    max_iter=max_iter)

            outputs = self.model(per_data)
            preds.append(outputs.detach())
            trues.append(batch_y)
            per_data_list.append(per_data.detach().cpu())
            initial_examples.append(batch_x.detach().cpu())

        preds = torch.cat(preds, 0).argmax(1)
        trues = torch.cat(trues, 0)
        per_data_list = torch.cat(per_data_list, 0)
        initial_examples = torch.cat(initial_examples, 0)
        m = metrics(trues.cpu().numpy(), preds.cpu().numpy())
        print_log(path_to_log, f'Epsilon: {epsilon} max_iter: {max_iter} {print_dict(**m)} lambda {lamb}, {add_info}\n')
        path_to_data = os.path.join(path, 'data')
        if not os.path.exists(path_to_data):
            os.mkdir(path_to_data)
        if attack == 'ifgsm_discr' or attack == 'deepfool_discr':
            path_to_data = os.path.join(path_to_data, f'{add_info}_{attack}_eps{epsilon}_mi{max_iter}_lam{lamb}')

        else: 
            path_to_data = os.path.join(path_to_data, f'{add_info}_{attack}_eps{epsilon}_mi{max_iter}')

        if not os.path.exists(path_to_data):
            os.mkdir(path_to_data)
        torch.save(per_data_list, os.path.join(path_to_data, f'per_data_list.pth'))
        torch.save(initial_examples, os.path.join(path_to_data, f'initial_data_list.pth'))
        print_log(path_to_log, f'Adversarial data and initial data are saved to {path_to_data}\n')

    def test_discr(self, path, discrim_critic, attack, epsilon, max_iter, discrim_adv = None, lamb = None):
        path_to_data = ''
        test_set, test_loader = get_dataset_loader(self.config, 'TEST')
        path_to_log = os.path.join(path, 'testing_adversarial_discriminator.txt')
        print_log(path_to_log, 'Loading model....\n')
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))
        print_log(path_to_log,'Loaded')
        print_log(path_to_log, "\n>>>>>>>>>>Testing_adversarial_with_dicriminator<<<<<<<<<<\n")
        preds = []
        trues = []
        preds_discr = []
        trues_discr = []
        per_data_list = []
        initial_examples = []
        attack_func = get_attack(attack)
        # set_model_to_eval_mode(self.model)
        discrim_critic.eval()
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_y_discr = torch.ones_like(batch_y).to(self.device)
            
            if attack == 'ifgsm_discr' or attack == 'deepfool_discr':
                per_data = attack_func(self.model, discrim_adv,
                                input=batch_x, 
                                target= batch_y,  
                                epsilon=epsilon, 
                                criterion=self.criterion, 
                                max_iter=max_iter, lamb = lamb)
            else:
                per_data = attack_func(self.model, 
                                    input=batch_x, 
                                    target= batch_y,  
                                    epsilon=epsilon, 
                                    criterion=self.criterion, 
                                    max_iter=max_iter)
                
        
            outputs = self.model(per_data)
            outputs_discr = discrim_critic(per_data)
            preds_discr.append(outputs_discr.detach())
            trues_discr.append(batch_y_discr.detach())
            preds.append(outputs.detach())
            trues.append(batch_y)
            per_data_list.append(per_data.detach().cpu())
            initial_examples.append(batch_x.detach().cpu())

        preds = torch.cat(preds, 0).argmax(1)
        trues = torch.cat(trues, 0)
        preds_discr = torch.cat(preds_discr, 0).argmax(1)
        print(preds_discr)
        trues_discr = torch.cat(trues_discr, 0)
        per_data_list = torch.cat(per_data_list, 0)
        initial_examples = torch.cat(initial_examples, 0)
        m = metrics(trues.cpu().numpy(), preds.cpu().numpy())
        acc_discr = calc_accuracy(trues_discr.cpu().numpy(), preds_discr.cpu().numpy())
        effect = 1 - m['accuracy_score']
        concealability = 1 - acc_discr
        print_log(path_to_log, f'Epsilon: {epsilon} max_iter: {max_iter} {print_dict(**m)} effectiveness={effect:.4f} concealability={concealability:.4f} lambda {lamb}\n')
        path_to_data = os.path.join(path, 'data')
        if not os.path.exists(path_to_data):
            os.mkdir(path_to_data)
        if attack == 'ifgsm_discr':
            path_to_data = os.path.join(path_to_data, f'{attack}_eps{epsilon}_mi{max_iter}_lam{lamb}')
        else:
            path_to_data = os.path.join(path_to_data, f'{attack}_eps{epsilon}_mi{max_iter}')
        if not os.path.exists(path_to_data):
            os.mkdir(path_to_data)
        torch.save(per_data_list, os.path.join(path_to_data, f'per_data_list.pth'))
        torch.save(initial_examples, os.path.join(path_to_data, f'initial_data_list.pth'))
        print_log(path_to_log, f'Adversarial data and initial data are saved to {path_to_data}\n')


    def visualize(self, true, advers = None, path='./pic', name = 'test.pdf', **kwargs):
            plt.style.use('stylesheet.mplstyle')
            plt.figure(figsize = (10, 5))
            plt.plot(true,  "--", linewidth = 1, label='Original data')
            if advers is not None:
                plt.plot(advers, label='Attacked data')

            plt.title(print_dict(**kwargs))
            plt.xlabel('Epsilon')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            if not os.path.exists(path):
                os.mkdir(path)

            plt.savefig(os.path.join(path, name))
        
        