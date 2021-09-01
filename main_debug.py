'''
Author: your name
Date: 2021-08-27 16:35:28
LastEditTime: 2021-09-01 10:12:48
LastEditors: your name
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
from custom_types import DaRnnNet, TrainData, TrainConfig
from constants import device
from  train import *
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")



'''
在这里初始化所有参数
'''

save_plots = True
debug = True
# TODO 1.读取数据集，如果是debug模式就读取前面100行 否则读取全部
raw_data = pd.read_excel(("data/dianli.xlsx"), nrows=1000 if debug else None)
logger.info(
    f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
raw_data_copy = raw_data.copy()
raw_data_copy.columns = ['date', 'hour',
                         'f1', 'f2', 'f3', 'f4', 'f5', 'target']
raw_data_copy = raw_data_copy[['f1', 'f2', 'f3', 'f4', 'f5', 'target']]
targ_cols = ("target",)  # NDX是我们需要预测的值
data, scaler = preprocess_data(raw_data_copy, targ_cols)

'''
指定参数
batch_size 128
T 10
learning_rate 0.001
n_epochs 30
hidden_size
'''


'''
初始化配置文件
'''

da_rnn_kwargs = {"batch_size": 128, "T": 10}
config, model = da_rnn(data, n_targs=len(targ_cols),
                       learning_rate=.001, **da_rnn_kwargs)

'''
定义损失函数
'''
iter_loss, epoch_loss = train(
    model, data, config, n_epochs=30, save_plots=save_plots)
del raw_data, raw_data_copy
