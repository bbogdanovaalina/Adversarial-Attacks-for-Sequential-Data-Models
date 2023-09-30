import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        bs, c, _ = x.size()
        
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)


class LSTMModel(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_layers, dropout_prob):
        
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = embed_dim
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, embed_dim, batch_first=True, dropout=dropout_prob,num_layers = num_layers
        )


    def forward(self, x):
        
        out, (hn, cn) = self.lstm(x)

        return out[:, -1, :]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, act_func = nn.ReLU):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding = kernel_size//2, dilation=dilation),
            act_func(),
            nn.BatchNorm1d(out_ch),
            SE_Block(out_ch)

        )
        self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):

        x = self.res_conv(x) + self.convs(x)

        return x


class CNN(nn.Module):

    def __init__(self, in_ch, out_ch, features = [128, 256, 128], kernel_size = 3, dilation = 1, act_func = nn.ReLU):
        super().__init__()

        self.convs = nn.ModuleList()

        for i in range(len(features)):
            self.convs.append(
                ConvBlock(in_ch, features[i], kernel_size=kernel_size, dilation=dilation, act_func=act_func)
            )

            in_ch = features[i]

        self.out = nn.Linear(features[-1], out_ch)


    def forward(self, x):

        x = x.permute(0, 2, 1)

        for conv in self.convs:

            x = conv(x)

        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.squeeze(dim=2)

        return self.out(x)
        


class MHA(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert (d_model % n_heads) == 0
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.dk = d_model // n_heads

        self.scale = self.dk ** -0.5

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, C = x.shape
        q = self.q(x).view(B, L, -1, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, -1, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, -1, self.dk).transpose(1, 2)

        att = q @ k.transpose(-2,-1) * self.scale

        att = torch.softmax(att, dim = -1)
        self.att = att
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, L, -1)
        out = self.out(out)

        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()

        self.mha = MHA(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):

        x = x + self.drop1(self.ln1(self.mha(x)))
        x = x + self.drop2(self.ln2(self.ff(x)))

        return x 
    

class Encoder(nn.Module):

    def __init__(self,  d_model, n_heads, n_layers, dropout = 0.1):
        super().__init__()
        self.net = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.cls = nn.Parameter(torch.randn(size = (1,1, d_model)))
        self.pe = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = self.pe(x)
        x = torch.cat([cls, x], dim=1)
        for layer in self.net:

            x = layer(x)

        return x[:, 0]
    


# class GenClass(nn.Module):

#     def __init__(self, in_ch, 
#                  embed_dim, 
#                  num_layers, 
#                  num_classes = 2,  
#                  dropout_prob = 0.1, 
#                  features = [128, 256, 128], 
#                  kernel_size = 3, 
#                  dilation = 1, 
#                  act_func = nn.ReLU, 
#                  num_heads=8, is_lstm = True):
#         super().__init__()
        
#         self.embedding = nn.Linear(in_ch, embed_dim)
        
#         self.first = LSTMModel(embed_dim, embed_dim, num_layers, dropout_prob) if is_lstm else Encoder(embed_dim, num_heads, num_layers, dropout_prob)

#         self.cnn = CNN(embed_dim, embed_dim, features = [128, 256, 128], kernel_size = 3, dilation = 1, act_func = nn.ReLU)

#         self.out = nn.Linear(embed_dim * 2, num_classes)
#     def forward(self, x):

#         x = self.embedding(x)

#         first = self.first(x)
#         cnn = self.cnn(x)
       
#         out = torch.cat([first, cnn], dim = 1)
        
        
#         return self.out(out)


class GenClass(nn.Module):

    def __init__(self, in_ch, d_model, models, num_classes):
        super().__init__()

        self.embed = nn.Linear(in_ch, d_model)
        self.models = nn.ModuleList([*models])

        self.fc = nn.Linear(d_model * len(models), num_classes)

    
    def forward(self, x):

        outputs = []

        x = self.embed(x)

        for model in self.models:
            outputs.append(model(x))

        outputs = torch.cat([*outputs], dim=1)

        return self.fc(outputs)