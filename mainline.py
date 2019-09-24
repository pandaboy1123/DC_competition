import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import re
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, BaggingRegressor,
                              )
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from demo import score
import logging
from keras import models
from keras import layers
from keras.optimizers import RMSprop

def conact_data(file_list):
    '''
    返回合并数据
    :param file_list:list
    :return:data
    '''
    data_list = [pd.read_csv(file_list[i], engine='python', encoding='utf-8') for i in range(len(file_list))]
    res_data = pd.concat(data_list)
    return res_data


def change_time_stamp(str):
    '''
    转换字符串时间为时间戳
    :param str:
    :return:
    '''
    return int(time.mktime(time.strptime(str, '%Y-%m-%d %X')))


def draw_pie(labels, quants, name):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # make a square figure
    plt.figure(1, figsize=(9, 9))
    colors = ["blue", "red", "coral", "green", "yellow", "orange"]  # 设置颜色（循环显示）
    plt.pie(quants, colors=colors, labels=labels, autopct='%3.2f%%', shadow=True)
    plt.title('%s' % name, bbox={'facecolor': '0.8', 'pad': 10})
    plt.show()
    plt.savefig("%s.png" % name)


def show_power(path):
    # path = r'E:\国能竞赛\demo\station_1_test.csv'
    with open(path, 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
        data.sort_values('时间', inplace=True)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        x = data['时间']
        y1 = data['预测功率']
        y2 = data['实际功率']
        plt.xlabel('time')
        plt.ylabel('power')
        plt.plot(x, y1, 'r', label='预测功率')
        plt.plot(x, y2, 'b', label='实际功率', alpha=0.7)
        plt.grid()
        plt.show()


def get_number(data):
    number = []
    # 实际功率小于0
    # print(len(real_p[real_p<0])/len(real_p))
    number.append(len(data[data < 0]))
    for i in range(0, 6):
        # print('实际功率大于等于%s且小于%s'%(i,i+1),len(real_p[(real_p>=i)&(real_p<(i+1))]))
        number.append(len(data[(data >= i) & (data < (i + 1))]))
    # 实际功率大于等于6
    number.append(len(data[data >= 6]))
    return number


def tmp_build_pie(train_data_1):
    '''
    临时用构造饼图
    :return:
    '''
    # 实际功率分布
    labels = []
    number = get_number(train_data_1['实际功率'])
    labels.append('小于0')
    for i in range(0, 6):
        labels.append('大于等于%s，小于%s' % (i, i + 1))
    labels.append('大于等于6')
    # draw_pie(labels,number,'功率分布图')


#####################根据装机容量进行分类################
# 装机容量10MW
#########################训练集###################
# path_1 = r'E:\国能竞赛\dataset\train\train_3.csv'
# path_2 = r'E:\国能竞赛\dataset\train\train_6.csv'
# file_list_1 = [path_1, path_2]
# train_data_1 = pd.DataFrame(conact_data(file_list_1))
# train_data_1['时间'] = [change_time_stamp(i) for i in train_data_1['时间']]
# train_data_1.sort_values("时间", inplace=True)
# # print(train_data_1.describe())
# #########################测试集###################
# test_path_1 = r'E:\国能竞赛\demo\3.csv'
# test_path_2 = r'E:\国能竞赛\demo\6.csv'
# test_file_list_1 = [test_path_1, test_path_2]
# test_data_1 = pd.DataFrame(conact_data(test_file_list_1))
# test_data_1['时间'] = [change_time_stamp(i) for i in test_data_1['时间']]
# print(test_data_1.head())
#####################xgboost####################
# X = train_data_1[['时间', '辐照度', '风速', '风向', '温度', '湿度', '压强']]
# y = train_data_1['实际功率']
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.4, random_state=666)
# X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.4, random_state=666)
# xgb_model = xgb.XGBRegressor(max_depth=8,max_delta_step=4,
#                              # min_child_weight=6,
#                              learning_rate=0.01,
#                              n_estimators=60000,
#                              objective='reg:squarederror',
#                              n_jobs=-1)
# xgb_model.fit(X_train, y_train,
#               eval_set=[(X_train, y_train)],
#               eval_metric='mae')
# y_pred = xgb_model.predict(X_test)
# y_test_pre = xgb_model.predict(test_data_1[['时间', '辐照度', '风速', '风向', '温度', '湿度', '压强']].values)
# pre_data = pd.DataFrame()
# pre_data['index'] = test_data_1['id']
# pre_data['prediction'] = y_test_pre
# pre_data.to_csv('pre_1.csv',index=False)
def use_xgboost(train_data, test_data, name, number, parm_list):
    X = train_data[['时间', '辐照度', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分']]
    y = train_data['实际功率']
    max_depth = parm_list[0]
    learning_rate = parm_list[1]
    n_estimators = parm_list[2]
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.4, random_state=666)
    X_train, X_test, y_train, y_test = train_test_split(
        X[['辐照度', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分']].values, y.values, test_size=0.4, random_state=666)
    xgb_model = xgb.XGBRegressor(max_depth=max_depth, max_delta_step=4,
                                 # min_child_weight=6,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 objective='reg:squarederror',
                                 n_jobs=-1)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train)],
                  eval_metric='mae')
    y_pred = xgb_model.predict(X_test)
    y_test_pre = xgb_model.predict(test_data[['辐照度', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分']].values)
    pre_data = pd.DataFrame()
    pre_data['index'] = test_data['id']
    pre_data['prediction'] = y_test_pre
    pre_data.to_csv('%s.csv' % name, index=False)
    new = pd.DataFrame()
    new['时间'] = X_test_1['时间']
    new['预测功率'] = y_pred
    new['实际功率'] = y_test
    new.to_csv('%s_test.csv' % name, index=False)
    # from demo import score
    # path = r'E:\国能竞赛\demo\%s_test.csv' % name
    # res = score.score(number, path)
    # print('%s综合评分为' % name, res)
    # show_power(path)


def K_fold(train_data, test_data, name, number, parm_list):
    X = train_data[
        ['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '实际功率', '辐照度^2', '辐照度 风速', '辐照度^3',
         '辐照度^2 风速',
         '辐照度 风速^2', '辐照度 风向^2',
         '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
         '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
         '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    # y = train_data['实际功率']
    y = test_data[
        ['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3', '辐照度^2 风速',
         '辐照度 风速^2',
         '辐照度 风向^2',
         '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
         '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
         '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    max_depth = parm_list[0]
    learning_rate = parm_list[1]
    n_estimators = parm_list[2]
    param = {
        'max_depth': max_depth,
        'max_delta_step': 4,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
    }
    dtrain = xgb.DMatrix(
        X[['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3',
           '辐照度^2 风速', '辐照度 风速^2', '辐照度 风向^2',
           '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
           '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
           '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']].values, label=X['实际功率'].values)
    dtest = xgb.DMatrix(y.values)
    cv_res = xgb.cv(param, dtrain, num_boost_round=4000, early_stopping_rounds=2000, nfold=10, metrics='mae',
                    verbose_eval=True,
                    )
    bst = xgb.train(param, dtrain, num_boost_round=cv_res.shape[0])
    y_pre = bst.predict(dtest)
    # print(len(y_pre),len(y))
    pre_data = pd.DataFrame()
    pre_data['index'] = test_data['id']
    pre_data['prediction'] = y_pre
    # print(pre_data.head())
    pre_data.to_csv('%s.csv' % name, index=False)


def log_in(name, res_score, bool):
    '''
    打印分数文件进入日志
    :param res_score:
    :return:
    '''
    if bool == 1:
        daylight = '白天'
    else:
        daylight = '黑夜'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [line:%(lineno)d]  %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=r'E:\国能竞赛\demo\MAE_score.log',
                        filemode='a+')
    logging.info('%s--%s的得分是%s' % (daylight, name, res_score))


def nothing_4_real(bst, X, name):
    x_pre = bst.predict(
        X[['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3',
           '辐照度^2 风速', '辐照度 风速^2', '辐照度 风向^2',
           '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
           '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
           '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']].values, num_iteration=bst.best_iteration)
    real_pre = X['实际功率'].values
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.hist(x_pre, bins=50, facecolor='blue', alpha=0.7, label='预测功率')
    plt.hist(real_pre, bins=50, facecolor='red', alpha=0.5, label='实际功率')
    plt.title('%s功率对比图' % name)
    plt.legend(loc='upper right')
    plt.show()
    return x_pre

def nothing_4_real_network(x_pre, y_pre, name):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.hist(x_pre.values, bins=50, facecolor='blue', alpha=0.7, label='预测功率')
    plt.hist(y_pre.values, bins=50, facecolor='red', alpha=0.5, label='实际功率')
    plt.title('%s功率对比图' % name)
    plt.legend(loc='upper right')
    plt.show()
    return x_pre


def lgb_Kfold(train_data, test_data, name, number, parm_list, bool):
    '''
    使用lgb进行交叉验证
    :param train_data:
    :param test_data:
    :param name:
    :param number:
    :param parm_list:
    :return:
    '''
    X = train_data[train_data['白天'] == bool][
        ['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '实际功率', '辐照度^2', '辐照度 风速', '辐照度^3',
         '辐照度^2 风速',
         '辐照度 风速^2', '辐照度 风向^2',
         '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
         '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
         '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    # y = train_data['实际功率']
    y = test_data[test_data['白天'] == bool][
        ['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3', '辐照度^2 风速',
         '辐照度 风速^2',
         '辐照度 风向^2',
         '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
         '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
         '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    max_depth = parm_list[0]
    learning_rate = parm_list[1]
    n_estimators = parm_list[2]
    param = {
        'max_depth': max_depth,
        'num_leaves': 16,
        'max_delta_step': 4,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'objective': 'regression',
        'n_jobs': -1,
    }
    dtrain = lgb.Dataset(
        X[['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3',
           '辐照度^2 风速', '辐照度 风速^2', '辐照度 风向^2',
           '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
           '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
           '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']].values, label=X['实际功率'].values)
    dtest = y.values
    lgb.cv(params=param, train_set=dtrain, stratified=False, num_boost_round=1000, early_stopping_rounds=2000, nfold=10,
           verbose_eval=True, metrics='mae')
    bst = lgb.train(param, dtrain)
    res = nothing_4_real(bst, X, name)
    score_data = pd.DataFrame()
    score_data['时间'] = train_data[train_data['白天'] == bool]['时间']
    score_data['实际功率'] = train_data[train_data['白天'] == bool]['实际功率']
    score_data['预测功率'] = res
    score_data.to_csv(r'E:\国能竞赛\dataset\score.csv', index=False)
    res_score = score.score(number, r'E:\国能竞赛\dataset\score.csv')
    log_in(name, res_score, bool)
    y_pre = bst.predict(dtest, num_iteration=bst.best_iteration)
    # print(y_pre)
    # print(len(y_pre),len(y))
    pre_data = pd.DataFrame()
    pre_data['index'] = test_data[test_data['白天'] == bool]['id']
    pre_data['prediction'] = y_pre
    # print(pre_data.head())
    pre_data.to_csv('%s.csv' % name, index=False)

def build_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = models.Sequential()
    model.add(layers.Dense(units=128, activation='relu', input_shape=(35, )))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=16, activation='relu'))
    '''
        网络的最后一层只有一个单元，没有激活，是一个线性层。这是标量回归（标量回归是预
    测单一连续值的回归）的典型设置。添加激活函数将会限制输出范围。例如，如果向最后一层
    添加 sigmoid 激活函数，网络只能学会预测 0~1 范围内的值。这里最后一层是纯线性的，所以
    网络可以学会预测任意范围内的值。
    '''
    model.add(layers.Dense(units=1))
    model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=['mae'])
    return model

def network_k_fold(train_data, test_data, name, number, parm_list, bool):
    k=10
    model = build_model()
    num_val_samples = len(train_data) // k
    X = train_data[train_data['白天'] == bool][
        ['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '实际功率', '辐照度^2', '辐照度 风速', '辐照度^3',
         '辐照度^2 风速',
         '辐照度 风速^2', '辐照度 风向^2',
         '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
         '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
         '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    # y = train_data['实际功率']
    y = test_data[test_data['白天'] == bool][
        ['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3', '辐照度^2 风速',
         '辐照度 风速^2',
         '辐照度 风向^2',
         '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
         '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
         '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    # dtrain = lgb.Dataset(
    #     X[['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3',
    #        '辐照度^2 风速', '辐照度 风速^2', '辐照度 风向^2',
    #        '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
    #        '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
    #        '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']].values, label=X['实际功率'].values)
    # dtest = y.values
    train = X[['总辐射', '直辐射', '散辐射', '风速', '风向', '温度', '湿度', '压强', '月', '日', '时', '分', '辐照度^2', '辐照度 风速', '辐照度^3',
           '辐照度^2 风速', '辐照度 风速^2', '辐照度 风向^2',
           '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3',
           '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2', '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2',
           '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']]
    targets = X[['实际功率']]
    test_data_y = y
    for i in range(k):
        print('正在进行第%s次验证'%(i+1))
        #  准备训练集，除了第i块都是训练集，这里用numpy的concatenate拼接
        partial_train_data = np.concatenate([train[:i*num_val_samples], train[(i+1)*num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([targets[:i * num_val_samples], targets[(i + 1) * num_val_samples:]],
                                            axis=0)
        '''
        verbose：日志显示
        verbose = 0 为不在标准输出流输出日志信息
        verbose = 1 为输出进度条记录
        verbose = 2 为每个epoch输出一行记录
        注意： 默认为 1
        '''
        model.fit(partial_train_data, partial_train_targets, epochs=100, batch_size=1, verbose=1)
    y_pre_one = model.predict(train)
    res_one = model.predict(test_data_y)
    score_data = pd.DataFrame()
    score_data['时间'] = train_data[train_data['白天'] == bool]['时间']
    score_data['实际功率'] = train_data[train_data['白天'] == bool]['实际功率']
    score_data['预测功率'] = np.array(y_pre_one).astype('float64')
    score_data.to_csv(r'E:\国能竞赛\dataset\score.csv', index=False)
    pre_data = pd.DataFrame()
    pre_data['index'] = test_data[test_data['白天'] == bool]['id']
    res_score = score.score(number, r'E:\国能竞赛\dataset\score.csv')
    log_in(name, res_score, bool)
    nothing_4_real_network(score_data['预测功率'], score_data['实际功率'], name)
    pre_data['prediction'] = np.array(res_one).astype('float64')
    pre_data.to_csv('%s.csv' % name, index=False)



# # 装机容量20MW
# #########################训练集###################
# path_3 = r'E:\国能竞赛\dataset\train\train_1.csv'
# path_4 = r'E:\国能竞赛\dataset\train\train_4.csv'
# path_5 = r'E:\国能竞赛\dataset\train\train_10.csv'
# file_list_2 = [path_3, path_4,path_5]
# train_data_2 = pd.DataFrame(conact_data(file_list_2))
# train_data_2['时间'] = [change_time_stamp(i) for i in train_data_2['时间']]
# train_data_2.sort_values("时间", inplace=True)
# #########################测试集###################
# test_path_3 = r'E:\国能竞赛\demo\1.csv'
# test_path_4 = r'E:\国能竞赛\demo\4.csv'
# test_path_5 = r'E:\国能竞赛\demo\10.csv'
# test_file_list_2 = [test_path_3, test_path_4, test_path_5]
# test_data_2 = pd.DataFrame(conact_data(test_file_list_2))
# # test_data_2['时间'] = [change_time_stamp(i) for i in test_data_2['时间']]
# # 装机容量21MW
# #########################训练集###################
# path_6 = r'E:\国能竞赛\dataset\train\train_5.csv'
# file_list_3 = [path_6]
# train_data_3 = pd.DataFrame(conact_data(file_list_3))
# train_data_3['时间'] = [change_time_stamp(i) for i in train_data_3['时间']]
# train_data_3.sort_values("时间", inplace=True)
# #########################测试集###################
# test_path_6 = r'E:\国能竞赛\dataset\test\test_5.csv'
# test_file_list_3 = [test_path_6]
# test_data_3 = pd.DataFrame(conact_data(test_file_list_3))
# test_data_3['时间'] = [change_time_stamp(i) for i in test_data_3['时间']]
# # 装机容量30MW
# #########################训练集###################
# path_7 = r'E:\国能竞赛\dataset\train\train_2.csv'
# path_8 = r'E:\国能竞赛\dataset\train\train_8.csv'
# file_list_4 = [path_7,path_8]
# train_data_4 = pd.DataFrame(conact_data(file_list_4))
# train_data_4['时间'] = [change_time_stamp(i) for i in train_data_4['时间']]
# train_data_4.sort_values("时间", inplace=True)
# #########################测试集###################
# test_path_7 = r'E:\国能竞赛\dataset\test\test_2.csv'
# test_path_8 = r'E:\国能竞赛\dataset\test\test_8.csv'
# test_file_list_4 = [test_path_7, test_path_8]
# test_data_4 = pd.DataFrame(conact_data(test_file_list_4))
# test_data_4['时间'] = [change_time_stamp(i) for i in test_data_4['时间']]
# # 装机容量40MW
# #########################训练集###################
# path_9 = r'E:\国能竞赛\dataset\train\train_7.csv'
# file_list_5 = [path_9]
# train_data_5 = pd.DataFrame(conact_data(file_list_5))
# train_data_5['时间'] = [change_time_stamp(i) for i in train_data_5['时间']]
# train_data_5.sort_values("时间", inplace=True)
# #########################测试集###################
# test_path_9 = r'E:\国能竞赛\dataset\test\test_7.csv'
# test_file_list_5 = [test_path_9]
# test_data_5 = pd.DataFrame(conact_data(test_file_list_5))
# test_data_5['时间'] = [change_time_stamp(i) for i in test_data_5['时间']]
# # 装机容量50MW
# #########################训练集###################
# path_10 = r'E:\国能竞赛\dataset\train\train_9.csv'
# file_list_6 = [path_10]
# train_data_6 = pd.DataFrame(conact_data(file_list_6))
# train_data_6['时间'] = [change_time_stamp(i) for i in train_data_6['时间']]
# train_data_6.sort_values("时间", inplace=True)
# #########################测试集###################
# test_path_10 = r'E:\国能竞赛\dataset\test\test_9.csv'
# test_file_list_6 = [test_path_10]
# test_data_6 = pd.DataFrame(conact_data(test_file_list_6))
# test_data_6['时间'] = [change_time_stamp(i) for i in test_data_6['时间']]
#####################根据装机容量进行分类################

# parm_list_2 = [16,0.01,60000]
# use_xgboost(train_data_2,test_data_2,'1and4and10',20,parm_list_2)
# parm_list_3 = [8,0.05,20000]
# use_xgboost(train_data_3,test_data_3,'5',21,parm_list_3)
# parm_list_4 = [8,0.05,20000]
# use_xgboost(train_data_4,test_data_4,'2and8',30,parm_list_4)
# parm_list_5 = [12,0.03,20000]
# use_xgboost(train_data_5,test_data_5,'7',40,parm_list_5)
# parm_list_6 = [16,0.01,80000]
# use_xgboost(train_data_6,test_data_6,'9',50,parm_list_6)

##################################按电站进行分类################
# 电站1
################################训练集###########################
path_1 = r'E:\国能竞赛\demo\train___1.csv'
file_list_1 = [path_1]
train_data_1 = pd.DataFrame(conact_data(file_list_1))
# print(train_data_1.head())
train_data_1['时间'] = [change_time_stamp(i) for i in train_data_1['时间']]
train_data_1.sort_values("时间", inplace=True)
################################测试集###########################
test_path_1 = r'E:\国能竞赛\demo\test___1.csv'
test_file_list_1 = [test_path_1]
test_data_1 = pd.DataFrame(conact_data(test_file_list_1))
# 电站2
################################训练集###########################
path_2 = r'E:\国能竞赛\demo\train___2.csv'
file_list_2 = [path_2]
train_data_2 = pd.DataFrame(conact_data(file_list_2))
train_data_2['时间'] = [change_time_stamp(i) for i in train_data_2['时间']]
train_data_2.sort_values("时间", inplace=True)
################################测试集###########################
test_path_2 = r'E:\国能竞赛\demo\test___2.csv'
test_file_list_2 = [test_path_2]
test_data_2 = pd.DataFrame(conact_data(test_file_list_2))
# 电站3
################################训练集###########################
path_3 = r'E:\国能竞赛\demo\train___3.csv'
file_list_3 = [path_3]
train_data_3 = pd.DataFrame(conact_data(file_list_3))
train_data_3['时间'] = [change_time_stamp(i) for i in train_data_3['时间']]
train_data_3.sort_values("时间", inplace=True)
################################测试集###########################
test_path_3 = r'E:\国能竞赛\demo\test___3.csv'
test_file_list_3 = [test_path_3]
test_data_3 = pd.DataFrame(conact_data(test_file_list_3))
# 电站4
################################训练集###########################
path_4 = r'E:\国能竞赛\demo\train___4.csv'
file_list_4 = [path_4]
train_data_4 = pd.DataFrame(conact_data(file_list_4))
train_data_4['时间'] = [change_time_stamp(i) for i in train_data_4['时间']]
train_data_4.sort_values("时间", inplace=True)
################################测试集###########################
test_path_4 = r'E:\国能竞赛\demo\test___4.csv'
test_file_list_4 = [test_path_4]
test_data_4 = pd.DataFrame(conact_data(test_file_list_4))
# 电站5
################################训练集###########################
path_5 = r'E:\国能竞赛\demo\train___5.csv'
file_list_5 = [path_5]
train_data_5 = pd.DataFrame(conact_data(file_list_5))
train_data_5['时间'] = [change_time_stamp(i) for i in train_data_5['时间']]
train_data_5.sort_values("时间", inplace=True)
################################测试集###########################
test_path_5 = r'E:\国能竞赛\demo\test___5.csv'
test_file_list_5 = [test_path_5]
test_data_5 = pd.DataFrame(conact_data(test_file_list_5))
# 电站6
################################训练集###########################
path_6 = r'E:\国能竞赛\demo\train___6.csv'
file_list_6 = [path_6]
train_data_6 = pd.DataFrame(conact_data(file_list_6))
train_data_6['时间'] = [change_time_stamp(i) for i in train_data_6['时间']]
train_data_6.sort_values("时间", inplace=True)
################################测试集###########################
test_path_6 = r'E:\国能竞赛\demo\test___6.csv'
test_file_list_6 = [test_path_6]
test_data_6 = pd.DataFrame(conact_data(test_file_list_6))
# 电站7
################################训练集###########################
path_7 = r'E:\国能竞赛\demo\train___7.csv'
file_list_7 = [path_7]
train_data_7 = pd.DataFrame(conact_data(file_list_7))
train_data_7['时间'] = [change_time_stamp(i) for i in train_data_7['时间']]
train_data_7.sort_values("时间", inplace=True)
################################测试集###########################
test_path_7 = r'E:\国能竞赛\demo\test___7.csv'
test_file_list_7 = [test_path_7]
test_data_7 = pd.DataFrame(conact_data(test_file_list_7))
# 电站8
################################训练集###########################
path_8 = r'E:\国能竞赛\demo\train___8.csv'
file_list_8 = [path_8]
train_data_8 = pd.DataFrame(conact_data(file_list_8))
train_data_8['时间'] = [change_time_stamp(i) for i in train_data_8['时间']]
train_data_8.sort_values("时间", inplace=True)
################################测试集###########################
test_path_8 = r'E:\国能竞赛\demo\test___8.csv'
test_file_list_8 = [test_path_8]
test_data_8 = pd.DataFrame(conact_data(test_file_list_8))
# 电站9
################################训练集###########################
path_9 = r'E:\国能竞赛\demo\train___9.csv'
file_list_9 = [path_9]
train_data_9 = pd.DataFrame(conact_data(file_list_9))
train_data_9['时间'] = [change_time_stamp(i) for i in train_data_9['时间']]
train_data_9.sort_values("时间", inplace=True)
################################测试集###########################
test_path_9 = r'E:\国能竞赛\demo\test___9.csv'
test_file_list_9 = [test_path_9]
test_data_9 = pd.DataFrame(conact_data(test_file_list_9))
# 电站10
################################训练集###########################
path_10 = r'E:\国能竞赛\demo\train___10.csv'
file_list_10 = [path_10]
train_data_10 = pd.DataFrame(conact_data(file_list_10))
train_data_10['时间'] = [change_time_stamp(i) for i in train_data_10['时间']]
train_data_10.sort_values("时间", inplace=True)
################################测试集###########################
test_path_10 = r'E:\国能竞赛\demo\test___10.csv'
test_file_list_10 = [test_path_10]
test_data_10 = pd.DataFrame(conact_data(test_file_list_10))
########################训练集###################
parm_list_1 = [6, 0.02, 2400]
# use_xgboost(train_data_1,test_data_1,'station_1',20,parm_list_1)
# K_fold(train_data_1, test_data_1, 'station_1', 20, parm_list_1)
# lgb_Kfold(train_data_1, test_data_1, 'station_1', 20, parm_list_1, 1)
# lgb_Kfold(train_data_1, test_data_1, 'station_1_1', 20, parm_list_1, 0)
network_k_fold(train_data_1, test_data_1, 'station_1', 20, parm_list_1, 1)
network_k_fold(train_data_1, test_data_1, 'station_1_1', 20, parm_list_1, 0)
parm_list_2 = [6, 0.02, 2400]
# # K_fold(train_data_2, test_data_2, 'station_2', 30, parm_list_2)
# # # use_xgboost(train_data_2,test_data_2,'station_2',30,parm_list_2)
# lgb_Kfold(train_data_2, test_data_2, 'station_2', 30, parm_list_2, 1)
# lgb_Kfold(train_data_2, test_data_2, 'station_2_2', 30, parm_list_2, 0)
network_k_fold(train_data_2, test_data_2, 'station_2', 30, parm_list_2, 1)
network_k_fold(train_data_2, test_data_2, 'station_2_2', 30, parm_list_2, 0)
parm_list_3 = [6, 0.02, 1000]
# # K_fold(train_data_3, test_data_3, 'station_3', 10, parm_list_3)
# # # use_xgboost(train_data_3,test_data_3,'station_3',10,parm_list_3)
# lgb_Kfold(train_data_3, test_data_3, 'station_3', 10, parm_list_3, 1)
# lgb_Kfold(train_data_3, test_data_3, 'station_3_3', 10, parm_list_3, 0)
network_k_fold(train_data_3, test_data_3, 'station_3', 10, parm_list_3, 1)
network_k_fold(train_data_3, test_data_3, 'station_3_3', 10, parm_list_3, 0)
parm_list_4 = [6, 0.02, 1000]
# # K_fold(train_data_4, test_data_4, 'station_4', 20, parm_list_4)
# # # use_xgboost(train_data_4,test_data_4,'station_4',20,parm_list_4)
# lgb_Kfold(train_data_4, test_data_4, 'station_4', 20, parm_list_4, 1)
# lgb_Kfold(train_data_4, test_data_4, 'station_4_4', 20, parm_list_4, 0)
network_k_fold(train_data_4, test_data_4, 'station_4', 20, parm_list_4, 1)
network_k_fold(train_data_4, test_data_4, 'station_4_4', 20, parm_list_4, 0)
parm_list_5 = [6, 0.01, 1000]
# # K_fold(train_data_5, test_data_5, 'station_5', 21, parm_list_5)
# # # use_xgboost(train_data_5,test_data_5,'station_5',21,parm_list_5)
# lgb_Kfold(train_data_5, test_data_5, 'station_5', 21, parm_list_5, 1)
# lgb_Kfold(train_data_5, test_data_5, 'station_5_5', 21, parm_list_5, 0)
network_k_fold(train_data_5, test_data_5, 'station_5', 21, parm_list_5, 1)
network_k_fold(train_data_5, test_data_5, 'station_5_5', 21, parm_list_5, 0)
parm_list_6 = [6, 0.01, 1000]
# # K_fold(train_data_6, test_data_6, 'station_6', 10, parm_list_6)
# # # use_xgboost(train_data_6,test_data_6,'station_6',10,parm_list_6)
# lgb_Kfold(train_data_6, test_data_6, 'station_6', 10, parm_list_6, 1)
# lgb_Kfold(train_data_6, test_data_6, 'station_6_6', 10, parm_list_6, 0)
network_k_fold(train_data_6, test_data_6, 'station_6', 10, parm_list_6, 1)
network_k_fold(train_data_6, test_data_6, 'station_6_6', 10, parm_list_6, 0)
parm_list_7 = [6, 0.01, 1000]
# # K_fold(train_data_7, test_data_7, 'station_7', 40, parm_list_7)
# # # use_xgboost(train_data_7,test_data_7,'station_7',40,parm_list_7)
# lgb_Kfold(train_data_7, test_data_7, 'station_7', 40, parm_list_7, 1)
# lgb_Kfold(train_data_7, test_data_7, 'station_7_7', 40, parm_list_7, 0)
network_k_fold(train_data_7, test_data_7, 'station_7', 40, parm_list_7, 1)
network_k_fold(train_data_7, test_data_7, 'station_7_7', 40, parm_list_7, 0)
parm_list_8 = [6, 0.01, 1000]
# # K_fold(train_data_8, test_data_8, 'station_8', 30, parm_list_8)
# # # use_xgboost(train_data_8,test_data_8,'station_8',30,parm_list_8)
# lgb_Kfold(train_data_8, test_data_8, 'station_8', 30, parm_list_8, 1)
# lgb_Kfold(train_data_8, test_data_8, 'station_8_8', 30, parm_list_8, 0)
network_k_fold(train_data_8, test_data_8, 'station_8', 30, parm_list_8, 1)
network_k_fold(train_data_8, test_data_8, 'station_8_8', 30, parm_list_8, 0)
parm_list_9 = [8, 0.01, 1000]
# # K_fold(train_data_9, test_data_9, 'station_9', 50, parm_list_9)
# # # use_xgboost(train_data_9,test_data_9,'station_9',50,parm_list_9)
# lgb_Kfold(train_data_9, test_data_9, 'station_9', 50, parm_list_9, 1)
# lgb_Kfold(train_data_9, test_data_9, 'station_9_9', 50, parm_list_9, 0)
network_k_fold(train_data_9, test_data_9, 'station_9', 50, parm_list_9, 1)
network_k_fold(train_data_9, test_data_9, 'station_9_9', 50, parm_list_9, 0)
parm_list_10 = [8, 0.01, 1000]
# # K_fold(train_data_10, test_data_10, 'station_10', 20, parm_list_10)
# # # use_xgboost(train_data_10,test_data_10,'station_10',20,parm_list_10)
# lgb_Kfold(train_data_10, test_data_10, 'station_10', 20, parm_list_10, 1)
# lgb_Kfold(train_data_10, test_data_10, 'station_10_10', 20, parm_list_10, 0)
network_k_fold(train_data_10, test_data_10, 'station_10', 20, parm_list_10, 1)
network_k_fold(train_data_10, test_data_10, 'station_10_10', 20, parm_list_10, 0)
