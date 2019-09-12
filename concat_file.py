# -*- encoding: utf-8 -*-
"""
@File    : concat_file.py
@Time    : 2019/9/7 13:18
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import seaborn as sns


def conact_data(file_list):
    '''
    返回合并数据
    :param file_list:list
    :return:data
    '''
    data_list = [pd.read_csv(file_list[i]) for i in range(len(file_list))]
    res_data = pd.concat(data_list)
    return res_data

path_1 = r'E:\国能竞赛\demo\station_1.csv'
path_2 = r'E:\国能竞赛\demo\station_2.csv'
path_3 = r'E:\国能竞赛\demo\station_3.csv'
path_4 = r'E:\国能竞赛\demo\station_4.csv'
path_5 = r'E:\国能竞赛\demo\station_5.csv'
path_6 = r'E:\国能竞赛\demo\station_6.csv'
path_7 = r'E:\国能竞赛\demo\station_7.csv'
path_8 = r'E:\国能竞赛\demo\station_8.csv'
path_9 = r'E:\国能竞赛\demo\station_9.csv'
path_10 = r'E:\国能竞赛\demo\station_10.csv'


file_list_1 = [path_1, path_2,path_3,path_4,path_5,path_6,path_7,path_8,path_9,path_10]
data = pd.DataFrame(conact_data(file_list_1))
print(data.head())
new_data = pd.DataFrame()
new_data['id'] = data['index']
new_data['prediction'] = data['prediction']
# print(len(new_data))
# print(data.sort_values('index',inplace=True).head())
new_data.to_csv('sample.csv',index=False)
