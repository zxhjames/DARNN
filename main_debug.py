# %%
import numpy as np

from lstm import *
from train import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")

'''
在这里初始化所有参数
'''

save_plots = True
debug = False
datasets = [
    "data/Austrilia_dianli.xlsx",
    "data/UCI_dianli.xlsx"
]

'''
读取数据集:澳大利亚数据集
'''
def read_dataset1(dataset,debug=True):
    raw_data = pd.read_excel((dataset), nrows=512 if debug else None)
    logger.info(
        f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
    raw_data_copy = raw_data.copy()
    raw_data_copy.columns = ['date', 'hour',
                             'f1', 'f2', 'f3', 'f4', 'f5', 'target']
    raw_data_copy = raw_data_copy[['f1', 'f2', 'f3', 'f4', 'f5', 'target']]
    targ_cols = ("target",)  # NDX是我们需要预测的值
    return raw_data_copy,targ_cols


'''
读取数据集:联合发电厂数据集
'''
def read_dataset2(dataset,debug=False):
    raw_data = pd.read_excel((dataset), nrows=1000 if debug else None)
    logger.info(
        f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
    raw_data_copy = raw_data.copy()
    raw_data_copy.columns = ['f1', 'f2', 'f3', 'f4', 'target']
    targ_cols = ("target",)  # NDX是我们需要预测的值
    return raw_data_copy, targ_cols


'''
数据预处理
'''
def prepare_data(dat, col_names):
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)
    #proc_dat = np.array(dat)
    # 生成同等列长的mask数组
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    return TrainData(feats, targs), scale

'''
读取数据集
'''
raw_data,targ_cols = read_dataset1(datasets[0])
data, scaler = prepare_data(raw_data, targ_cols)

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
        print('model',model)
        iter_loss, epoch_loss = train_rnn(
            model,data,config,n_epochs=30,save_plots=save_plots)

    # elif mode_name == 'Seq2Seq':
    #     s2s_args = init_args
    #     config, model = seq2seq(data, n_targs=len(targ_cols), learning_rate=.001, **s2s_args)
    #     print('model', model)
    #     iter_loss, epoch_loss = train_s2s(
    #         model, data, config, n_epochs=30, save_plots=save_plots)


trainMode('RNN')

'''
定义损失函数
'''

del raw_data, targ_cols
