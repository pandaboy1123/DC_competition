# -*- encoding: utf-8 -*-
"""
@File    : 结果融合.py
@Time    : 2019/9/23 8:31
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np

path_1 = r'E:\国能竞赛\dataset\new_dataset\气象数据\1569381042000sample.csv'
path_2 = r'E:\国能竞赛\dataset\new_dataset\气象数据\1569460729000sample.csv'
# path_3 = r'E:\国能竞赛\demo\1569307387000sample.csv'
with open(path_1,'r',encoding='utf-8') as f1:
    data = pd.read_csv(f1)
with open(path_2,'r',encoding='utf-8') as f2:
    data_2 = pd.read_csv(f2)
# with open(path_3,'r',encoding='utf-8') as f3:
#     data_3 = pd.read_csv(f3)
new_data = pd.merge(data,data_2,how='left',on='id')
# new_data = pd.merge(new_data,data_3,how='left',on='id')
print(new_data.head())
new_data['prediction'] = new_data.apply(lambda x: (0.12*x['prediction_x'] + x['prediction_y']*0.88), axis=1)
# print(new_data[['id','prediction']].head())
new_data[['id','prediction']].to_csv('pandasample.csv',index=False)