# -*- encoding: utf-8 -*-
"""
@File    : 特征工程.py
@Time    : 2019/9/17 16:46
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def poly_f(path, name):
    '''
    新构建的特征组合
    :param path:
    :param name:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
        poly_train = data[['辐照度', '风速', '风向', '温度', '湿度', '压强']]
        poly_transformer = PolynomialFeatures(degree=4)
        poly_transformer.fit(poly_train)
        poly_train = poly_transformer.transform(poly_train)
        feature_name = poly_transformer.get_feature_names(input_features=['辐照度', '风速', '风向', '温度', '湿度', '压强', ])
        poly_features = pd.DataFrame(poly_train,
                                     columns=poly_transformer.get_feature_names(['辐照度', '风速', '风向', '温度', '湿度', '压强']))
        name_li = ['辐照度^2', '辐照度 风速', '辐照度^3', '辐照度^2 风速', '辐照度 风速^2', '辐照度 风向^2', '风速^2 湿度', '辐照度 温度^2',
                   '辐照度 湿度^2', '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3', '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2',
                   '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2', '辐照度^4', '辐照度^3 风速', '风向^3 湿度', '风速 温度^3']
        data_need = poly_features[name_li]
        new_data = pd.concat([data, data_need], axis=1)
        print(new_data.head())
        new_data.to_csv('%s.csv' % name, index=False)



# test_path = [r'E:\国能竞赛\demo\test__1.csv', r'E:\国能竞赛\demo\test__2.csv', r'E:\国能竞赛\demo\test__3.csv',
#              r'E:\国能竞赛\demo\test__4.csv',
#              r'E:\国能竞赛\demo\test__5.csv', r'E:\国能竞赛\demo\test__6.csv', r'E:\国能竞赛\demo\test__7.csv',
#              r'E:\国能竞赛\demo\test__8.csv',
#              r'E:\国能竞赛\demo\test__9.csv', r'E:\国能竞赛\demo\test__10.csv']
# test_name = [r'test___1', r'test___2', r'test___3', r'test___4', r'test___5', r'test___6', r'test___7', r'test___8',
#              r'test___9', r'test___10']
# train_path = [r'E:\国能竞赛\demo\train__1.csv', r'E:\国能竞赛\demo\train__2.csv', r'E:\国能竞赛\demo\train__3.csv',
#               r'E:\国能竞赛\demo\train__4.csv',
#               r'E:\国能竞赛\demo\train__5.csv', r'E:\国能竞赛\demo\train__6.csv', r'E:\国能竞赛\demo\train__7.csv',
#               r'E:\国能竞赛\demo\train__8.csv',
#               r'E:\国能竞赛\demo\train__9.csv', r'E:\国能竞赛\demo\train__10.csv']
# train_name = [r'train___1', r'train___2', r'train___3', r'train___4', r'train___5', r'train___6', r'train___7',
#               r'train___8', r'train___9', r'train___10']
# for i in range(len(test_path)):
#     poly_f(test_path[i],test_name[i])
#     poly_f(train_path[i],train_name[i])
path = r'E:\国能竞赛\demo\train___1.csv'
data = pd.read_csv(path)
print(data.columns.values)