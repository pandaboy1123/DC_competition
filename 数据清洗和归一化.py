# -*- encoding: utf-8 -*-
"""
@File    : 数据清洗和归一化.py
@Time    : 2019/9/11 11:02
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_the_number(path, number):
    '''
    删除训练集合的脏数据
    :param path:
    :param number:
    :return:
    '''
    # path = r'E:\国能竞赛\dataset\new_dataset\train\train_1.csv'
    with open(path,'r',encoding='utf-8') as f:
        train_data = pd.read_csv(f)
        print(train_data['实际辐照度'].describe())
        print(len(train_data[train_data['实际辐照度']>=number]))
        print(len(train_data[train_data['实际辐照度'] < 0]))
        new_data = train_data[(train_data['实际辐照度']<number)&(train_data['实际辐照度']>=0)]
    x = np.array(new_data['实际辐照度'])
    y = np.array(new_data['实际功率'])
    plt.scatter(x,y,alpha=0.6)
    plt.show()
    return new_data

path = r'E:\国能竞赛\dataset\new_dataset\train\train_10.csv'
number = 2500
data = pd.DataFrame(find_the_number(path,number))
data.to_csv('train_10_10.csv',index=False)

