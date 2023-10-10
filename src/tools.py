import os
import torch
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
def print_dict(**kwargs):
    result = ""
    for key, value in kwargs.items():
        result += f"{key}={value} "
    return result.strip()

def metrics(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    ras = roc_auc_score(true, pred)

    return {
        'accuracy_score': acc,
        'f1_score': f1,
        'roc_auc_score': ras
    }

def set_lr(optim, lr):
    for p in optim.param_groups:
        p['lr'] = lr
    print(f'Lr is adjusted to {lr}')

def cosine_annealing_lr(epoch, num_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))

def visual_data(true, advers = None, figsize = (12, 7), path='./pic', name = 'test.pdf', **kwargs):
    """
    Results visualization
    """
    plt.figure(figsize=figsize)
    plt.plot(true, label='GroundTruth', linewidth=2)
    if advers is not None:
        plt.plot(advers, label='Adversarial', linewidth=2)
    plt.legend()
    plt.title(print_dict(**kwargs))
    plt.grid()
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(os.path.join(path, name), bbox_inches='tight')

@dataclass
class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss





        


