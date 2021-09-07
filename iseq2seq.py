import torch.nn as nn
import torch
from sklearn.metrics.mape import mean_absolute_percentage_error
from torch.nn import functional as F
from custom_types import *
from train import prep_train_data,adjust_learning_rate
from torch import optim
import utils
import json
from constants import device
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import numpy_to_tvar
from sklearn.metrics import mean_squared_error,mean_absolute_error
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src): 
        '''
        src = [src_len, batch_size]
        '''
        # src = src.transpose(0, 1) # src = [batch_size, src_len]
        # embedded = self.dropout(self.embedding(src)).transpose(0, 1) # embedded = [src_len, batch_size, emb_dim]
        
        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(src) # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer
        
        # enc_hidden [-2, :, : ] is the last of the forwards RNN 
        # enc_hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2,:,:], enc_hidden[-1,:,:]), dim = 1)))
        
        return enc_output, s



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, s, enc_output):
        
         # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]
        
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)
        
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)
        

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        #self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, s, enc_output):
             
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        
        #embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]
        
        # a = [batch_size, 1, src_len]  
        a = self.attention(s, enc_output).unsqueeze(1)
        
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        # rnn_input = torch.cat((embedded, c), dim = 2)
        rnn_input = torch.cat((c), dim = 2)  
        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        #embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        
        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c), dim = 1))
        
        return pred, dec_hidden.squeeze(0)


'''
模型的初始化
'''
def seq2seq(train_data: TrainData, n_targs: int, hidden_size=64, T=10, learning_rate=0.001, batch_size=128):
    
    # 定义配置器 T=>滑窗长度 截取前70%的数据作为训练集
    train_cfg = TrainConfig(
        T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")
    # 初始化网络结构
    rnn_args = {
        "input_size" : train_data.feats.shape[1] + 1,
        "hidden_size" : hidden_size,
        "num_layers" : 1,
        "output_size" : 1,
        "dropout" : 0,
        "batch_first":True
    }
   # print ("run args: ", rnn_args)
    rnn  = Encoder(**rnn_args).to(device)
    with open( ('data/lstm.json'),"w") as fi: 
        json.dump(rnn_args,fi,indent=4)

    params=[p for p in rnn.parameters() if p.requires_grad]
   # print('params: ',params)
    rnn_optimizer = optim.Adam(
        params=params,
        lr=learning_rate
    )

    # 返回的网络结构
    rnn_net = RnnNet(
        rnn,rnn_optimizer
    )
    return train_cfg, rnn_net
    