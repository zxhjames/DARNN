import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.bidirectional = False
        self.lstm_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1,bidirectional=self.bidirectional)
        self.attn_linear = nn.Linear(in_features=1 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        # cell = init_hidden(input_data, self.hidden_size)

        '''
        input_weighted 34 9 5
        input encoder 34 9 64
        hidden 1 34 64
        cell 1 34 64
        input_data 34 9 5
        '''
      # (batch_size, input_size)
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            # TODO permute转化tensor维度 encoder的输入是隐藏状态和每一个cell
            #  repeat表示对张量进行扩充，在这里是对通道数(行)进扩充
            #  cat是对tensor进行拼接

            # todo
            # hidden repeat 34 5 64
            # cell repeat 34 5 64
            # input_data 34 5 9
            
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                        # cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                        input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)

            # todo x 34 5 137

            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 1 + self.T - 1))  # (batch_size * input_size) * 1
            # todo x (170 1)
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)
            # todo 34 5
            # Eqn. 10: LSTM 逐元素相乘
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # todo 34 5
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden))
            # 1 34 5 ,1 34 64, 1 34 64
            hidden = lstm_states
            #cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.bidirectional = False
        self.attn_layer = nn.Sequential(nn.Linear(1 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.GRU(input_size=out_feats, hidden_size=decoder_hidden_size,bidirectional=self.bidirectional)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        #cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                          # cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),z
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x =   self.attn_layer(x.view(-1, 1 * self.decoder_hidden_size + self.encoder_hidden_size))

            x = tf.softmax(x.view(-1, self.T - 1),dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden))
            hidden = lstm_output  # 1 * batch_size * decoder_hidden_size
            #cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))