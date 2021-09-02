'''
Author: your name
Date: 2021-08-31 10:06:33
LastEditTime: 2021-09-02 08:58:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/研二/code/lstm.py
'''
from torch.autograd.variable import Variable
import torch.nn as nn 
import torch
from custom_types import *
from train import prep_train_data,adjust_learning_rate
from torch import optim
import utils
import json
from constants import device
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import numpy_to_tvar
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")

'''
初始化简单的lstm网络
'''
class Lstm(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers , output_size , dropout, batch_first):
        super(Lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first

        self.rnn = nn.LSTM(input_size=self.input_size, 
                           hidden_size=self.hidden_size, 
                           num_layers=self.num_layers, 
                           batch_first=self.batch_first, 
                           dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        return out




'''
基本网络类型
'''
def rnn(train_data: TrainData, n_targs: int, hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):

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
    rnn  = Lstm(**rnn_args).to(device)
    with open( ('data/lstm.json'),"w") as fi:
        json.dump(rnn_args,fi,indent=4)

    rnn_optimizer = optim.Adam(
        params=[p for p in rnn.parameters() if p.requires_grad],
        lr=learning_rate
    )

    # 返回的网络结构
    rnn_net = RnnNet(
        rnn,rnn_optimizer
    )
    return train_cfg, rnn_net
    


'''
准备训练数据 对于每一组数据，需要分离特征( 历史前T时间步的基本特征和标签作为训练集，下一步作为测试集)
'''
def prep_rnn_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    print('batch_ids: ',batch_idx)
    # batch_idx是一个随机索引的下标
    # 特征数据 feats shape (128,9,5)
    feats = np.zeros((len(batch_idx), t_cfg.T , train_data.feats.shape[1]))
    # 历史序列 y_history (128,9,1)
    y_history = np.zeros((len(batch_idx), t_cfg.T , train_data.targs.shape[1]))
    # 标签数据 y_target (128,1)
    y_target = train_data.targs[batch_idx + t_cfg.T]


    # 获取采样的batch_id的下标和值
    # 获取特征和标签的相应下标值
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T )
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target



'''
用于rnn计算预测值与损失值的函数
'''
def rnn_train_iteration(t_net: RnnNet, loss_func: typing.Callable, X, y_history, y_target):
    input_data = np.append(X,y_history,axis=2)
    # print(input_data.shape)
    data1 = torch.from_numpy(input_data).to(torch.float32)
    pred = t_net.rnn(Variable(data1))
    pred = pred[0, :, :]
    label = torch.from_numpy(y_target).to(torch.float32).unsqueeze(1)
    loss = loss_func(pred, label)
    t_net.rnn_optimizer.zero_grad()
    loss.backward()
    t_net.rnn_optimizer.step()
    print('loss计算完成')
    return loss.item()



'''
rnn 版本预测
'''
def rnn_predict(t_net: RnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(
            input_encoded, y_history).cpu().data.numpy()

    return y_pred



'''
DaRnnNet 网络架构
train_data 训练集
t_cfg 训练集配置
n_epochs 迭代次数
save_plots 是否保存图片
'''
def train_rnn(net: RnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    # 向上取整
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    # 存储迭代损失值
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    # 存储每轮epoch的损失值
    epoch_losses = np.zeros(n_epochs)
    logger.info(
        f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")
    n_iter = 0
    print('一共', n_epochs, '次迭代')
    for e_i in range(n_epochs):
        print('现在是第 ', e_i, '轮迭代')
        # 随机生成
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        # 循环迭代 每次迭代的步长为batch_size\
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            # 随机采样
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            # 滑窗策略

            # 相当于 X,
            feats, y_history, y_target = prep_rnn_train_data(
                batch_idx, t_cfg, train_data)

            # 计算loss值
            loss = rnn_train_iteration(net, t_cfg.loss_func,
                                   feats, y_history, y_target)
            print('loss: ',loss)
                                   
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        print('epoch_loss',epoch_losses)
        # if e_i % 10 == 0:
        #     y_test_pred = rnn_predict(net, train_data,
        #                           t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
        #                           on_train=False)
        #     # TODO: make this MSE and make it work for multiple inputs
        #     val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
        #     logger.info(
        #         f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
        #     y_train_pred = rnn_predict(net, train_data,
        #                            t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
        #                            on_train=True)
        #     plt.figure()
        #     plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
        #              label="True")
        #     plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
        #              label='Predicted - Train')
        #     plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
        #              label='Predicted - Test')
        #     plt.legend(loc='upper left')
        #     utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses
