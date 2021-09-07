'''
todo 训练过程
'''
import numpy as np
from sklearn.metrics.mape import mean_absolute_percentage_error
from DARNN.custom_types import DaRnnNet, TrainData, TrainConfig
from sklearn.metrics import mean_squared_error,mean_absolute_error
from DARNN.train import adjust_learning_rate
from utils import *

def simple_train_darnn(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    # 向上取整
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    # 存储迭代损失值
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    # 存储每轮epoch的损失值
    epoch_losses = np.zeros(n_epochs)
    # print('iter_loss:', iter_losses)
    # print('epoch_loss: ', epoch_losses)
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

            print(feats.shape,y_history.shape,y_target.shape)
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