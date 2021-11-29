import torch
import torch.nn as nn
import torch.nn.functional as F

class WSCNNLSTM(nn.Module):
    def __init__(self, len: int):
        super(WSCNNLSTM, self).__init__()
        
        self.window = 8
        self.seqlen = 31
        
        self.kmer = 3
        self.dim = 4 ** self.kmer
        self.RNN_hidden = 128
        self.dropout_rate = 0.3

        
        self.ConvLayer = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=1, padding="same"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d()
        )
        
        self.gru = nn.GRU(input_size = 31, hidden_size = 8, num_layers = 2, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flattening = nn.Flatten()
        
        self.predictor = nn.Sequential(
            nn.Linear(in_features=2304, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=256, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, inputs):
        
        xlist = list()
        for i in range(self.seqlen - (self.window - 1)):
            window_seq = inputs[:,i:i+self.window]
            window_seq = F.one_hot(window_seq, num_classes=self.dim).to(torch.float)
            xlist.append(window_seq)
            
        x = torch.stack(xlist, dim=1) #256, 24, 8, 64
        x = self.ConvLayer(x) #256, 48, 3, 31
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))    
        
        xout, _ = self.gru(x)
        F_RNN = xout[:,:,:self.RNN_hidden]
        R_RNN = xout[:,:,self.RNN_hidden:]
        xout = torch.cat((F_RNN, R_RNN), 2)
        xout = self.dropout(xout)
        
        xout = self.flattening(xout)
        xout = self.predictor(xout)
        return xout.squeeze()
        