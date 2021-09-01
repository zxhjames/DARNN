'''
Author: your name
Date: 2021-08-26 10:55:50
LastEditTime: 2021-09-01 14:46:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/研二/code/custom_types.py
'''
import collections
import typing

import numpy as np


class TrainConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable


class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray


DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])

RnnNet = collections.namedtuple("RnnNet",["rnn","rnn_optimizer"])
