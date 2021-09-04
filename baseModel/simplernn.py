# %%
import pandas as pd
import numpy as np

save_plot = True
debug = True
data_dir = '../data/dianli.xlsx'
data = pd.read_excel(data_dir, nrows=1000 if debug else None)

# %%

raw_data_copy = data.copy()
raw_data_copy.columns = ['date', 'hour',
                         'f1', 'f2', 'f3', 'f4', 'f5', 'target']
raw_data_copy = raw_data_copy[['f1', 'f2', 'f3', 'f4', 'f5', 'target']]
targ_cols = ("target",)  # NDX是我们需要预测的值


# %%

# 数据预处理
class TrainConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable


class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray


from sklearn.preprocessing import StandardScaler


def read2Dataloader(data, T, batchSize):
    l = len(data.columns) - 1
    scale = StandardScaler().fit(data)
    df = (scale.transform(data))
    Y = df[:, l:]
    X = df[:, 0:l]
    return TrainData(X, Y)


# 返回特征
trainData = read2Dataloader(raw_data_copy, 10, 128)

# %%

trainData.targs.shape

# %%

'''
基本网络类型
'''
import torch.nn as nn
import json
from torch import optim
import collections
import typing

'''
初始化简单的lstm网络
'''
RnnNet = collections.namedtuple("RnnNet", ["rnn", "rnn_optimizer"])


class Lstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, batch_first):
        super(Lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=self.batch_first,
                          dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(
            x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        return out


def rnn(train_data: TrainData, n_targs: int, hidden_size: int, T: int, learning_rate=0.001, batch_size=128):
    # 定义配置器 T=>滑窗长度 截取前70%的数据作为训练集
    train_cfg = TrainConfig(
        T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    print('train size: ', train_cfg.train_size)
    input_size = train_data.feats.shape[1]
    print('加1是为了增加历史列 input size: ', input_size)
    # 初始化网络结构
    rnn_args = {
        "input_size": input_size + 1,
        "hidden_size": hidden_size,
        "num_layers": 1,
        "output_size": 1,
        "dropout": 0,
        "batch_first": True
    }
    print("run args: ", rnn_args)
    rnn = Lstm(**rnn_args)
    with open(('../data/lstm.json'), "w") as fi:
        json.dump(rnn_args, fi, indent=4)

    rnn_optimizer = optim.Adam(
        params=rnn.parameters(),
        lr=learning_rate
    )
    # 返回的网络结构
    rnn_net = RnnNet(
        rnn, rnn_optimizer
    )
    return train_cfg, rnn_net


# %%

# 初始化模型参数
init_args = {"batch_size": 128, "T": 10}
rnn_kwargs = init_args
config, model = rnn(data, n_targs=len(targ_cols), hidden_size=64, T=10, learning_rate=.001, batch_size=128)

# %%

# 平滑处理序列 开始训练
import torch
from torch.autograd import Variable


def PrepareData(batch_idx, t_cfg, train_data):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    print('y history shape', y_history.shape)
    print('train data targs  shape', train_data.targs.shape)
    # 获取采样的batch_id的下标和值
    # 获取特征和标签的相应下标值
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        # print('b_i',b_i,'b_slc',b_slc)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        # print(y_history[b_i : ].shape,train_data.targs[b_slc].shape)
        y_history[b_i:] = train_data.targs[b_slc]

    return feats, y_history, y_target


def train_iteration(t_net: RnnNet, loss_func: typing.Callable, X, y_history, y_target):
    input_data = np.append(X, y_history, axis=2)
    data1 = torch.from_numpy(input_data).to(torch.float32)
    print('input shape', data1.shape)
    pred = t_net.rnn(Variable(data1))
    print('pred shape: ', pred.shape)
    pred = pred[0, :, :]
    label = torch.from_numpy(y_target).to(torch.float32).unsqueeze(1)
    loss = loss_func(pred, label)
    t_net.rnn_optimizer.zero_grad()
    loss.backward()
    t_net.rnn_optimizer.step()
    return loss.item()


def train_rnn(net: RnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs, save_plots=False):
    # 完成所有数据训练的迭代次数
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    print('iter_per_epoch: ', t_cfg.train_size, t_cfg.batch_size, iter_per_epoch)
    # 存储损失值列表
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    print('iter_losses: ', iter_losses.shape)
    # 存储每次epoch的损失值
    epoch_losses = np.zeros(n_epochs)

    n_iter = 0
    print('一共', n_epochs, '次迭代')

    for e_i in range(n_epochs):
        print('现在是第 ', e_i, '轮迭代')
        # 随机生成
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)

        # 循环迭代 每次迭代的步长为batch_size，每次选择batch_size大小的数据进行预测
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            # 随机选择索引列
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]

            # TODO 闭包函数，返回处理好的特征，历史数据，预测目标值

            feats, y_history, y_target = PrepareData(batch_idx, t_cfg, train_data)
            print('feats', feats.shape)
            print('y_history', y_history.shape)
            print('y_target', y_target.shape)
            loss = train_iteration(net, t_cfg.loss_func,
                                   feats, y_history, y_target)
            print('loss', loss)

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        print('epoch_loss', epoch_losses)
    print('训练结束')
    return iter_losses, epoch_losses


iter_loss, epoch_loss = train_rnn(
    model, data, config, n_epochs=30, save_plots=True)

# %%
