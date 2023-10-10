import torch
import torch.nn as nn
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


class LSTMCell(nn.Module):

    def __init__(self, in_ch, out_ch, bidirectional = True, dropout = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(
                in_ch, hidden_size =out_ch, batch_first=True, bidirectional = bidirectional
            )
        # self.bn = nn.BatchNorm1d(out_ch * (2 if bidirectional else 1))
        mul = 2 if bidirectional else 1
        self.ln = nn.LayerNorm(out_ch * mul)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(mul * out_ch, out_ch)
        self.bd = bidirectional
    def forward(self, x):
        
        x, (_,_) = self.lstm(x)
        
        x = self.ln(x)
        if self.bd:
            x = self.fc(x)
        x = self.drop(x)
        return x


class LSTMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_ch = config.in_ch
        h_sizes = config.h_sizes
        bidirectional = config.bidirectional
        dropout=config.dropout
        # num_classes = config.num_classes

        self.net = nn.ModuleList()

        for i in range(len(h_sizes)):

            self.net.append(
                LSTMCell(in_ch, h_sizes[i], bidirectional, dropout)
            )

            in_ch = h_sizes[i]

        # self.fc = nn.Linear(h_sizes[-1], d_model)

    def forward(self, x):
        print(x.shape)
        for net in self.net:
            x = net(x)
        # x = self.fc(x[:, -1, : ])
        return x[:, -1, : ]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, act_func = nn.ReLU):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding = 'same', dilation=dilation),
            SE_Block(out_ch),
            
            nn.BatchNorm1d(out_ch),
            act_func(),
            

        )
        self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):

        x = self.convs(x) + self.res_conv(x)

        return x
        
class CNNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_ch = config.in_ch 
        features = config.features
        kernel_size = config.kernel_size 
        dilation = config.dilation 
        act_func = config.act_func 
        
        self.convs = nn.ModuleList()

        for i in range(len(features)):
            self.convs.append(
                ConvBlock(in_ch, features[i], kernel_size=kernel_size, dilation=dilation, act_func=act_func)
            )

            in_ch = features[i]
        self.drop = nn.Dropout(0.1)
        # self.out = nn.Linear(features[-1], num_classes)


    def forward(self, x):

        x = x.permute(0, 2, 1)

        for conv in self.convs:

            x = self.drop(conv(x))

        x = torch.mean(x, dim = -1)

        return x

class MHA(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert (d_model % n_heads) == 0

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

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
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, mul = 4, dropout = 0.1):
        super().__init__()

        self.mha = MHA(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mul * d_model),
            nn.GELU(),
            nn.Linear(mul*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):

        x = x + self.drop1(self.ln1(self.mha(x)))
        x = x + self.drop2(self.ln2(self.ff(x)))

        return x 
    

class Encoder(nn.Module):

    def __init__(self,  config):
        super().__init__()

        d_model = config.d_model
        n_heads = config.n_heads
        n_layers = config.n_layers
        dropout = config.dropout
        mul = config.mul

        self.net = nn.ModuleList(
            [EncoderLayer(d_model = d_model, n_heads = n_heads, mul=mul, dropout=dropout) for _ in range(n_layers)]
        )
        # self.cls = nn.Parameter(torch.randn(size = (1,1, d_model)))
        self.pe = PositionalEncoding(d_model, dropout)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # cls = self.cls.expand(x.shape[0], -1, -1)
        x = self.pe(x)
        # x = torch.cat([cls, x], dim=1)
        for layer in self.net:

            x = layer(x)
        x = x.permute(0,2,1).mean(dim = -1)
        return x
    

    