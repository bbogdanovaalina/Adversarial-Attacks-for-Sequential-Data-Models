import torch
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_from_tsfile
from sklearn.model_selection import train_test_split
import pandas as pd

class FordADataset(Dataset):
    def __init__(self, config, flag):
        if flag == 'TRAIN':
            self.X_data, self.y_data = load_from_tsfile(
                config.root_path + "/FordA_TRAIN.ts", return_data_type = "numpy3d"
            )
        elif flag == 'TEST':
            self.X_data, self.y_data = load_from_tsfile(
                config.root_path + "/FordA_TEST.ts", return_data_type = "numpy3d"

            )
        
        self.y_data = self.y_data .astype(int)

        self.y_data[self.y_data == -1] = 0
        
        self.y_data = torch.from_numpy(self.y_data).long()
        self.X_data = torch.from_numpy(self.X_data).float()
        
    def __getitem__(self, index):
        x = self.X_data[index].unsqueeze(-1).squeeze(0)
        y = self.y_data[index]

        return x, y
        
    def __len__ (self):
        return len(self.X_data)
    

class ChinatownDataset(Dataset):
    def __init__(self, config, flag):
        if flag == 'TRAIN':
            self.X_data, self.y_data = load_from_tsfile(
                config.root_path + "/Chinatown_TRAIN.ts", return_data_type = "numpy3d"
            )
        elif flag == 'TEST':
            self.X_data, self.y_data = load_from_tsfile(
                config.root_path + "/Chinatown_TEST.ts", return_data_type = "numpy3d"
            )
        
        self.y_data = self.y_data .astype(int)

        self.y_data[self.y_data == 2] = 0
        
        self.y_data = torch.from_numpy(self.y_data).long()
        self.X_data = torch.from_numpy(self.X_data).float()
        
    def __getitem__(self, index):
        x = self.X_data[index].unsqueeze(-1).squeeze(0)
        y = self.y_data[index]

        return x, y
        
    def __len__ (self):
        return len(self.X_data)
    

class WineDataset(Dataset):
    def __init__(self, config, flag):
        if flag == 'TRAIN':
            self.X_data, self.y_data = load_from_tsfile(
                config.root_path + "/Wine_TRAIN.ts", return_data_type = "numpy3d"
            )
        elif flag == 'TEST':
            self.X_data, self.y_data = load_from_tsfile(
                config.root_path + "/Wine_TEST.ts", return_data_type = "numpy3d"
            )
        
        self.y_data = self.y_data .astype(int)

        self.y_data[self.y_data == 2] = 0
        self.y_data = torch.from_numpy(self.y_data).long()
        self.X_data = torch.from_numpy(self.X_data).float()
        
    def __getitem__(self, index):
        
        x = self.X_data[index].unsqueeze(-1).squeeze(0)
        y = self.y_data[index]

        return x, y
        
    def __len__ (self):
        return len(self.X_data)

class FordADiscrDataset(Dataset):
    def __init__(self, config, flag):
        df = pd.read_csv(config.root_path, sep = '\t')
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy().astype(int) 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0xC0FFEE)
        if flag == 'TRAIN':
            self.X_data, self.y_data  = X_train, y_train
        elif flag == 'TEST':
            self.X_data, self.y_data = X_test, y_test

        self.X_data = torch.from_numpy(self.X_data).float()
        self.y_data = torch.from_numpy(self.y_data).long()
        self.max_seq_len = self.X_data.shape[1]
        self.feature_df = torch.randn(size=(1, 1))
        self.class_names = [0, 1]

    def __getitem__(self, index):
        x = self.X_data[index].unsqueeze(-1)
        y = self.y_data[index]
        return x, y

    def __len__(self):
        return len(self.X_data)

datasets = {
    'FordA': FordADataset,
    'WineDataset': WineDataset,
    'ChinatownDataset': ChinatownDataset,
    'FordADiscr': FordADiscrDataset
}

def get_dataset_loader(config, flag):
    
    dataset = datasets[config.dataset_name](config, flag = flag)
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, 
    num_workers=config.num_workers)

    return dataset, loader

