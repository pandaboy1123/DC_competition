# -*- encoding: utf-8 -*-
"""
@File    : score.py
@Time    : 2019/9/5 14:26
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import time
import datetime
import re
from functools import reduce

# 设置装机容量
ci = 10
# 设置文件路径
path = r'E:\国能竞赛\demo\test.csv'

def score(ci,path):
    '''
    :param ci:设置装机容量
    :param a_0:设置比较参数，这个参数用来比较实际功率
    :param path:设置文件路径
    :return:
    '''
    # 设置比较参数，这个参数用来比较实际功率
    a_0 = int(ci) * 0.03
    with open(path, 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
        # 判断实际功率是否大于装机容量功率
        data = data[data['实际功率'] >= a_0]
        if data.empty:
            mean_value = 99
        else:
            data.sort_values('时间', inplace=True)
            data['时间'] = [datetime.datetime.fromtimestamp(i) for i in data['时间']]
            data['时间'] = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in data['时间']]
            day_list = list(data['时间'].drop_duplicates())
            month_list = list([time.strftime("%Y-%m", time.strptime(i, '%Y-%m-%d')) for i in data['时间'].drop_duplicates()])
            func = lambda x, y: x if y in x else x + [y]
            f = reduce(func, [[], ] + month_list)
            day_dict = {}
            month_dict = {}
            MAE_D = {}
            MAE_M = {}
            # print('开始运算.....')
            for day in day_list:
                day_dict[day] = len(data[data['时间'].str.contains(day)])
            for month in month_list:
                month_dict[month] = month_list.count(month)
            data['score'] = data.apply(lambda x: abs(x['实际功率'] - x['预测功率']), axis=1)
            for k,v in day_dict.items():
                MAE_D[k] = (((sum(data[data['时间'].str.contains(k)].score))/ci)/day_dict[k])
            new_data = pd.DataFrame.from_dict(MAE_D,orient='index',columns=['score'])
            new_data = new_data.reset_index().rename(columns={'index':'time'})
            for k,v in month_dict.items():
                MAE_M[k] = (sum(new_data[new_data['time'].str.contains(k)].score)/len(new_data[new_data['time'].str.contains(k)]))
            new_data_1 = pd.DataFrame.from_dict(MAE_M, orient='index', columns=['score'])
            new_data_1 = new_data_1.reset_index().rename(columns={'index': 'time'})
            mean_value = (np.mean(new_data_1.score))
        return mean_value


print(score(10,r'E:\国能竞赛\dataset\score.csv'))