import torch
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_from_tsfile

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
    

datasets = {
    'FordA': FordADataset,
    'WineDataset': WineDataset,
    'ChinatownDataset': ChinatownDataset
}

def get_dataset_loader(config, flag):
    
    dataset = datasets[config.dataset_name](config, flag = flag)
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, 
    num_workers=config.num_workers)

    return dataset, loader

