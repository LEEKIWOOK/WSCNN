import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """
    def __init__(self,in_features, hidden_units,num_task=1):
        super(BahdanauAttention,self).__init__()
        self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

        score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        #print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights

class HLAttn(nn.Module):
    def __init__(self, len: int):
        super(HLAttn, self).__init__()
        
        self.kmer = 3
        self.dim = 4 ** self.kmer
        self.RNN_hidden = 100
        self.dropout_rate = 0.3
        
        self.embedding_layer = nn.Embedding(self.dim, 128, max_norm = True)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.Bilstm = nn.LSTM(input_size = 128, hidden_size = 128, batch_first=True, bidirectional=True)
        self.Attention = BahdanauAttention(in_features = 256, hidden_units = self.RNN_hidden)

        self.predictor = nn.Sequential(
            nn.Linear(in_features=256, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        x = self.dropout(self.embedding_layer(inputs)) #256, 31, 256
        
        output,(h_n,c_n) = self.Bilstm(x)
        h_n = h_n.view(batch_size,output.size()[-1])
        context_vector,attention_weights = self.Attention(h_n,output)
        
        xout = self.predictor(context_vector.squeeze())
        return xout.squeeze()
        