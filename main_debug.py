'''
Author: your name
Date: 2021-08-27 16:35:28
LastEditTime: 2021-09-01 18:17:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/研二/code/main_debug.py
'''
# %%
import typing
from typing import Tuple
import json
import os
import torch
from torch import nn
from torch import optim
import joblib
import pandas as pd
import numpy as np

import utils
from custom_types import *
from constants import device
from lstm import * 
from  train import * 
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")



'''
在这里初始化所有参数
'''

save_plots = True
debug = True
raw_data = pd.read_excel(("data/dianli.xlsx"), nrows=1000 if debug else None)
logger.info(
    f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
raw_data_copy = raw_data.copy()
raw_data_copy.columns = ['date', 'hour',
                         'f1', 'f2', 'f3', 'f4', 'f5', 'target']
raw_data_copy = raw_data_copy[['f1', 'f2', 'f3', 'f4', 'f5', 'target']]
targ_cols = ("target",)  # NDX是我们需要预测的值


'''
数据预处理
'''
def prepare_data(dat, col_names):
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)
    # 生成同等列长的mask数组
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    return TrainData(feats, targs), scale
    
data, scaler = prepare_data(raw_data_copy, targ_cols)


# 获取模型选项并训练模型
init_args = {"batch_size": 128, "T": 10}

def trainMode(mode_name):
    if mode_name == 'DARNN':
        da_rnn_kwargs = init_args
        config, model = da_rnn(data, n_targs=len(targ_cols),
                            learning_rate=.001, **da_rnn_kwargs)
        print('model',model)
        iter_loss, epoch_loss = train(
            model, data, config, n_epochs=30, save_plots=save_plots)


    elif mode_name == 'RNN':
        rnn_args = init_args
        config, model = rnn(data, n_targs=len(targ_cols),learning_rate=.001,**rnn_args)
        print('model',model.rnn)
        iter_loss, epoch_loss = train_rnn(
            model,data,config,n_epochs=30,save_plots=save_plots)

    


trainMode('RNN')

'''
定义损失函数
'''

del raw_data, raw_data_copy
