# -*- encoding: utf-8 -*-
"""
@File    : id检查.py
@Time    : 2019/9/26 16:31
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
import pandas as pd
path = r'E:\国能竞赛\demo\1569381042000sample.csv'
path_1 = r'E:\国能竞赛\demo\sample.csv'
with open(path,'r',encoding='utf-8') as f:
    data = pd.read_csv(f)
with open(path_1,'r',encoding='utf-8') as f:
    data2 = pd.read_csv(f)
new = pd.merge(data,data2,how='left',on='id')
print(len(new),len(data),len(data2))
new = new.drop_duplicates('id')
new['prediction'] = new.apply(lambda x: (0.5*x['prediction_x'] + x['prediction_y']*0.5), axis=1)
new[['id','prediction']].to_csv('psample.csv',index=False)

print(len(new))