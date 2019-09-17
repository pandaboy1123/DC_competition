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

path = r'E:\国能竞赛\demo\train__1.csv'
with open(path,'r',encoding='utf-8') as f:
    data = pd.read_csv(f)
    poly_train  = data[['辐照度','风速','风向','温度','湿度','压强']]
    # print(data.columns.values)
    poly_transformer = PolynomialFeatures(degree=4)
    poly_transformer.fit(poly_train)
    poly_train = poly_transformer.transform(poly_train)
    feature_name = poly_transformer.get_feature_names(input_features=['辐照度','风速','风向','温度','湿度','压强',])[:15]

    # 新建一个数据框存放这些特征
    poly_features = pd.DataFrame(poly_train,
                                 columns=poly_transformer.get_feature_names(['辐照度','风速','风向','温度','湿度','压强']))
    # '年', '月', '日', '时', '分'
    # print(poly_features.head())
    features_number = len(poly_features.columns.values)
    poly_features['实际功率'] = data['实际功率']
    # print(poly_features.columns.values)
    print('构造了%s个特征数' % features_number)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    li = [poly_features.columns.values[i:i+8] for i in range(1,len(poly_features.columns.values),8)]
    for i in li:
        new_i = list(i)
        new_i.append('实际功率')
        corr = abs(poly_features[new_i].corr())
        fig = plt.figure()
        fig.add_subplot(figsize=(20, 20))
        sns.heatmap(corr, linewidths=0.05, vmax=1, vmin=0, annot=True,
                         annot_kws={'size': 6, 'weight': 'bold'})
        plt.xticks(np.arange(len(new_i)) + 0.5, new_i)
        plt.yticks(np.arange(len(new_i)) + 0.5, new_i)
        plt.show()