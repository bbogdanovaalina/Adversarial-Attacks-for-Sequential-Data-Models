import torch
import torch.nn as nn
from .modules import CNNLayer, LSTMLayer, Encoder


class LSTM(nn.Module):
    
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.lstm = LSTMLayer(config)
        self.fc = nn.Linear(config.h_sizes[-1], config.num_classes)

    def forward(self, x):
        
        x = self.lstm(x)
        x = self.fc(x)
        return x




class CNN(nn.Module):

    def __init__(self, config):
        super().__init__()
    
        self.cnn = CNNLayer(config)
        self.out = nn.Linear(config.features[-1], config.num_classes)


    def forward(self, x):

        x = self.cnn(x)

        return self.out(x)
    
        
class TranClass(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.fc = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):

        x = self.encoder(x)
        x = self.fc(x)

        return x
    

class LSTM_FCN(nn.Module):

    def __init__(self, config):

        self.cnn = CNNLayer(config)
        self.lstm = LSTMLayer(config)

        self.out = nn.Linear(config.h_sizes[-1] + config.features[-1], config.num_classes)
    
    def forward(self, x):

        x = torch.cat([self.cnn(x), self.lstm(x)], dim = 1)

        return self.out(x)


models = {
    'CNN': CNN,
    'TranClass': TranClass,
    'LSTM_FCN': LSTM_FCN,
    'LSTM': LSTM
}
def get_model(model_name):

    return models[model_name]