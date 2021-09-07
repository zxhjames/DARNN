'''
Author: your name
Date: 2021-09-01 10:08:37
LastEditTime: 2021-09-02 13:49:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/研二/code/train.py
'''


# %%
import typing
from typing import Tuple
import json

from sklearn.metrics.mape import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import optim
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from imodules import Encoder, Decoder
from custom_types import *
from utils import numpy_to_tvar
from constants import device

import utils
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")
from sklearn.metrics import *

def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.001, batch_size=128):

    # 定义配置器 T=>滑窗长度 截取前70%的数据作为训练集
    train_cfg = TrainConfig(
        T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")
    enc_kwargs = {
        "input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)

    # 将encoder层，decoder层配置写 入配置文件
    with open(("data/enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)
    with open(("data/enc_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(
        encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


'''
TODO 原论文中的训练方法
DaRnnNet 网络架构
train_data 训练集
t_cfg 训练集配置
n_epochs 迭代次数
save_plots 是否保存图片
'''


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    # 向上取整
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    # 存储迭代损失值
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    # 存储每轮epoch的损失值
    epoch_losses = np.zeros(n_epochs)
    # print('iter_loss:', iter_losses)
    # print('epoch_loss: ', epoch_losses)
    logger.info(
        f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

   # print('一共', n_epochs, '次迭代')
    for e_i in range(n_epochs):
        #print('现在是第 ', e_i, '轮迭代')

        # 随机生成
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        # 循环迭代 每次迭代的步长为batch_size\
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            # todo 随机采样 batch_idx的长度为128个
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            #print('perm id ' ,perm_idx)
            #print('batch id: ',batch_idx)
            # 滑窗策略

            feats, y_history, y_target = prep_train_data(
                batch_idx, t_cfg, train_data)

            #print(feats.shape,y_history.shape,y_target.shape)
            # 计算loss值

            loss = train_iteration(net, t_cfg.loss_func,
                                   feats, y_history, y_target)

            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

       # print('epoch_loss',epoch_losses)

        if e_i % 10 == 0:
            print('开始预测')
            y_test_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logger.info(
                f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                   on_train=True)
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc='upper left')

            print(1,1 + len(train_data.targs))
            print(t_cfg.T,len(y_train_pred) + t_cfg.T)
            print(t_cfg.T + len(y_train_pred),len(train_data.targs) +1)


            # todo 计算三者最后的MSE MAE MAPE
            y_test_list = list(y_test_pred)
            y_real = list(train_data.targs)[t_cfg.T + len(y_train_pred)-1 :len(train_data.targs) ]
            print(len(y_real),len(y_test_list))


            print('rmse: ',np.sqrt(mean_squared_error(y_real,y_test_list)))
            print('mae: ', mean_absolute_error(y_real, y_test_list))
            print('mape: ', mean_absolute_percentage_error(y_real, y_test_list))
            utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses


'''
准备训练数据
'''
def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros(
        (len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target

'''
自适应调整学习率
'''
def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9

'''
DARNN训练迭代
'''
def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))
   # print('y_pred shape: ',y_pred.shape)
    y_true = numpy_to_tvar(y_target)
    #print('y_true shape: ',y_true.shape)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()





'''
预测代码
'''
def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
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


