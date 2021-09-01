'''
Author: your name
Date: 2021-08-31 10:06:33
LastEditTime: 2021-09-01 13:30:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/研二/code/lstm.py
'''
import torch.nn as nn 

'''
初始化简单的lstm网络
'''
class Lstm(nn.Module):
    
    def __init__(self, input_size:int, hidden_size:int, T:int):
        super(Lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size # 隐藏层个数
        self.input_size = input_size # 输入大小
        self.num_layers = 1 # 层数
        self.output_size = 1
        self.dropout = 0
        self.batch_first = False
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        return out


    