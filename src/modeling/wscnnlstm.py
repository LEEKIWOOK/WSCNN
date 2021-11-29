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
            nn.Conv1d(8, 16, 3, stride=1, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(),
        )
        
        self.gru = nn.GRU(input_size = 31, hidden_size = 8, num_layers = 2, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flattening = nn.Flatten()
        
        self.predictor = nn.Sequential(
            nn.Linear(in_features=2688, out_features=256),
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
            window_seq = self.ConvLayer(window_seq)
            
            window_seq, _ = self.gru(window_seq)
            F_RNN = window_seq[:,:,:self.RNN_hidden]
            R_RNN = window_seq[:,:,self.RNN_hidden:]
            xout = torch.cat((F_RNN, R_RNN), 2)
            
            xout = self.dropout(xout)
            xout = self.maxpool(xout)
            xlist.append(xout)
        
        x = torch.cat(xlist, dim=1)
        x = self.flattening(x)
        x = self.predictor(x)
        return x.squeeze()
        