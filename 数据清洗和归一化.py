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
import time
import os

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

# path = r'E:\国能竞赛\dataset\new_dataset\train\train_10.csv'
# number = 2500
# data = pd.DataFrame(find_the_number(path,number))
# data.to_csv('train_10_10.csv',index=False)
################################数据归一化############################3
def stand_2_one(path,name):
    '''
    数据归一化
    :param path:
    :return:
    '''
    # path = r'E:\国能竞赛\demo\1.csv'
    with open(path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        # print(data['实际辐照度'].describe())
        # print(len(data[data['实际辐照度']<0]))
        data.loc[data['实际辐照度'][(data['实际辐照度']<0)].index,['实际辐照度']] = 0
        data['归一化实际辐照度'] = [((i-min(data['实际辐照度']))/(max(data['实际辐照度']-min(data['实际辐照度'])))) for i in data['实际辐照度']]
        # print(data['实际辐照度'].head())
        # print(data['实际辐照度'].describe())
        data.to_csv('%s.csv'%name,index=False)
        print('%s已经保存完成！'%name)
    return data

# test_path_1 = r'E:\国能竞赛\demo\1.csv'
# test_path_2 = r'E:\国能竞赛\demo\2.csv'
# test_path_3 = r'E:\国能竞赛\demo\3.csv'
# test_path_4 = r'E:\国能竞赛\demo\4.csv'
# test_path_5 = r'E:\国能竞赛\demo\5.csv'
# test_path_6 = r'E:\国能竞赛\demo\6.csv'
# test_path_7 = r'E:\国能竞赛\demo\7.csv'
# test_path_8 = r'E:\国能竞赛\demo\8.csv'
# test_path_9 = r'E:\国能竞赛\demo\9.csv'
# test_path_10 = r'E:\国能竞赛\demo\10.csv'
# stand_2_one(test_path_1,'test_1')
# stand_2_one(test_path_2,'test_2')
# stand_2_one(test_path_3,'test_3')
# stand_2_one(test_path_4,'test_4')
# stand_2_one(test_path_5,'test_5')
# stand_2_one(test_path_6,'test_6')
# stand_2_one(test_path_7,'test_7')
# stand_2_one(test_path_8,'test_8')
# stand_2_one(test_path_9,'test_9')
# stand_2_one(test_path_10,'test_10')
#
#
# train_path_1 = r'E:\国能竞赛\dataset\train\train_1.csv'
# train_path_2 = r'E:\国能竞赛\dataset\train\train_2.csv'
# train_path_3 = r'E:\国能竞赛\dataset\train\train_3.csv'
# train_path_4 = r'E:\国能竞赛\dataset\train\train_4.csv'
# train_path_5 = r'E:\国能竞赛\dataset\train\train_5.csv'
# train_path_6 = r'E:\国能竞赛\dataset\train\train_6.csv'
# train_path_7 = r'E:\国能竞赛\dataset\train\train_7.csv'
# train_path_8 = r'E:\国能竞赛\dataset\train\train_8.csv'
# train_path_9 = r'E:\国能竞赛\dataset\train\train_9.csv'
# train_path_10 = r'E:\国能竞赛\dataset\train\train_10.csv'
# stand_2_one(train_path_1,'train1')
# stand_2_one(train_path_2,'train2')
# stand_2_one(train_path_3,'train3')
# stand_2_one(train_path_4,'train4')
# stand_2_one(train_path_5,'train5')
# stand_2_one(train_path_6,'train6')
# stand_2_one(train_path_7,'train7')
# stand_2_one(train_path_8,'train8')
# stand_2_one(train_path_9,'train9')
# stand_2_one(train_path_10,'train10')

# with open(r'E:\国能竞赛\demo\train1.csv','r',encoding='utf-8') as f:
#     data = pd.read_csv(f)
#     print(data.head())
#     for i in data['时间']:
#         if i =='0':
#             print(i)
#         else :
#             print('一切正常')

#test测试集时间转换
def data_2_part(path,name):
    with open(path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        data['年'] = [time.strftime("%Y", time.localtime(i)) for i in data['时间']]
        data['月'] = [time.strftime("%m", time.localtime(i)) for i in data['时间']]
        data['日'] = [time.strftime("%d", time.localtime(i)) for i in data['时间']]
        data['时'] = [time.strftime("%H", time.localtime(i)) for i in data['时间']]
        data['分'] = [time.strftime("%M", time.localtime(i)) for i in data['时间']]
        data.to_csv('%s.csv'%name,index=False)
        print('%s保存完成'%name)


def train_2_date(path,name):
    with open(path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        for i in data['时间']:
            data['年'] = time.strftime("%Y", time.localtime(int(time.mktime(time.strptime(i, '%Y-%m-%d %X')))))
            data['月'] = time.strftime("%m", time.localtime(int(time.mktime(time.strptime(i, '%Y-%m-%d %X')))))
            data['日'] = time.strftime("%d", time.localtime(int(time.mktime(time.strptime(i, '%Y-%m-%d %X')))))
            data['时'] = time.strftime("%H", time.localtime(int(time.mktime(time.strptime(i, '%Y-%m-%d %X')))))
            data['分'] = time.strftime("%M", time.localtime(int(time.mktime(time.strptime(i, '%Y-%m-%d %X')))))
        data.to_csv('%s.csv'%name,index=False)
        print('%s保存完成'%name)


# #################测试集转换###############
# path_1 = r'E:\国能竞赛\demo\test_1.csv'
# path_2 = r'E:\国能竞赛\demo\test_2.csv'
# path_3 = r'E:\国能竞赛\demo\test_3.csv'
# path_4 = r'E:\国能竞赛\demo\test_4.csv'
# path_5 = r'E:\国能竞赛\demo\test_5.csv'
# path_6 = r'E:\国能竞赛\demo\test_6.csv'
# path_7 = r'E:\国能竞赛\demo\test_7.csv'
# path_8 = r'E:\国能竞赛\demo\test_8.csv'
# path_9 = r'E:\国能竞赛\demo\test_9.csv'
# path_10 = r'E:\国能竞赛\demo\test_10.csv'
# data_2_part(path_1,'test__1')
# data_2_part(path_2,'test__2')
# data_2_part(path_3,'test__3')
# data_2_part(path_4,'test__4')
# data_2_part(path_5,'test__5')
# data_2_part(path_6,'test__6')
# data_2_part(path_7,'test__7')
# data_2_part(path_8,'test__8')
# data_2_part(path_9,'test__9')
# data_2_part(path_10,'test__10')
# ################训练集转换###############
# t_path_1 = r'E:\国能竞赛\demo\train1.csv'
# t_path_2 = r'E:\国能竞赛\demo\train2.csv'
# t_path_3 = r'E:\国能竞赛\demo\train3.csv'
# t_path_4 = r'E:\国能竞赛\demo\train4.csv'
# t_path_5 = r'E:\国能竞赛\demo\train5.csv'
# t_path_6 = r'E:\国能竞赛\demo\train6.csv'
# t_path_7 = r'E:\国能竞赛\demo\train7.csv'
# t_path_8 = r'E:\国能竞赛\demo\train8.csv'
# t_path_9 = r'E:\国能竞赛\demo\train9.csv'
# t_path_10 = r'E:\国能竞赛\demo\train10.csv'
# train_2_date(t_path_1,'train__1')
# train_2_date(t_path_2,'train__2')
# train_2_date(t_path_3,'train__3')
# train_2_date(t_path_4,'train__4')
# train_2_date(t_path_5,'train__5')
# train_2_date(t_path_6,'train__6')
# train_2_date(t_path_7,'train__7')
# train_2_date(t_path_8,'train__8')
# train_2_date(t_path_9,'train__9')
# train_2_date(t_path_10,'train__10')



def read_file(path):
    '''
    读取文件的行数
    :param path:
    :return:
    '''
    with open(path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        print(len(data))

# with open(train_path,'r',encoding='utf-8') as f:
#     train_data = pd.read_csv(f)
#     # print(train_data.columns.values)
# with open(real_path,'r',encoding='utf-8') as f1:
#     real_data = pd.read_csv(f1)
#     # print(real_data.columns.values)
#     new_data = pd.merge(train_data,real_data,how='left',on='时间')
#     # print(new_data.head())
#     print(new_data.columns.values,len(new_data),len(train_data))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.hist(new_data['压强_x'],bins=50,alpha=0.9,facecolor='red')
# plt.hist(new_data['压强_y'],bins=50,alpha=0.6,facecolor='blue')
# plt.show()
# sample_path_1 = r'E:\国能竞赛\1568548639000sample.csv'
# sample_path_2 = r'E:\国能竞赛\1568876391000sample.csv'
# with open(sample_path_1,'r',encoding='utf-8') as f:
#     data_1 = pd.read_csv(f)
# with open(sample_path_2,'r',encoding='utf-8') as f1:
#     data_2 = pd.read_csv(f1)
#     new_data = pd.merge(data_2,data_1,how='left',on='id')
#     new_data['prediction'] = new_data.apply(lambda x: (x['prediction_x']*0.8 + x['prediction_y']*0.2), axis=1)
#     data = new_data[['id','prediction']]
#     data.to_csv('sample1.csv',index=False)
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     plt.hist(data_1['prediction'],bins=50,facecolor='blue',alpha=0.7)
#     plt.hist(data_2['prediction'],bins=50,facecolor='red',alpha=0.6)
#     plt.hist(new_data['prediction'],bins=50,facecolor='yellow',alpha=0.6)
#     plt.show()


def change_time_stamp(str):
    '''
    转换字符串时间为时间戳
    :param str:
    :return:
    '''
    return int(time.mktime(time.strptime(str, '%Y-%m-%d %X')))


def rebuild_file(train_path,power_path):
    '''
    根据实际数据进行修正
    :param path:
    :return:
    '''
    with open(train_path,'r',encoding='utf-8') as f:
        train_data = pd.read_csv(f)
        # train_data['时间'] = [time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(i)) for i in train_data['时间']]
    with open(power_path,'r',encoding='utf-8') as f1:
        power_data = pd.read_csv(f1)
    new_data = pd.merge(train_data,power_data,how='left',on='时间')
    new_data = new_data.drop(columns=['辐照度', '风速_x', '风向_x', '温度_x', '湿度_x', '压强_x'], axis=1)
    new_data.rename(columns={'风速_y': '风速', '风向_y': '风向', '温度_y': '温度', '湿度_y': '湿度', '压强_y': '压强'}, inplace=True)
    new_data.to_csv(train_path,index=False)
    print(train_path+'保存完成')

# path_1 = r'E:\国能竞赛\demo\test___1.csv'
# path_2 = r'E:\国能竞赛\demo\test___2.csv'
# path_3 = r'E:\国能竞赛\demo\test___3.csv'
# path_4 = r'E:\国能竞赛\demo\test___4.csv'
# path_5 = r'E:\国能竞赛\demo\test___5.csv'
# path_6 = r'E:\国能竞赛\demo\test___6.csv'
# path_7 = r'E:\国能竞赛\demo\test___7.csv'
# path_8 = r'E:\国能竞赛\demo\test___8.csv'
# path_9 = r'E:\国能竞赛\demo\test___9.csv'
# path_10 = r'E:\国能竞赛\demo\train___10.csv'
# path_li = [path_1,path_2,path_3,path_4,path_5,path_6,path_7,path_8,path_9,path_10]
# p_1 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站1_气象.csv'
# p_2 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站2_气象.csv'
# p_3 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站3_气象.csv'
# p_4 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站4_气象.csv'
# p_5 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站5_气象.csv'
# p_6 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站6_气象.csv'
# p_7 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站7_气象.csv'
# p_8 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站8_气象.csv'
# p_9 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站9_气象.csv'
# p_10 = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站10_气象.csv'

# p_list = [p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10]
# for i in range(len(path_li)):
#     rebuild_file(path_li[i],p_list[i])

def stand_2_one_2(path_li,str_li):
    '''
    数据归一化
    :param str_li:输入指定字段进行归一化
    :return:
    '''
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    for a in range(len(path_li)):
        with open(path_li[a],'r',encoding='utf-8') as f:
            data = pd.read_csv(f)
            print(path_li[a]+'文件已打开')
            for x in range(len(str_li)):
                print(x,str_li[x])
                data[[str_li[x]]].apply(max_min_scaler)
                print(path_li[a]+str_li[x]+'修改完成')
            data.to_csv(path_li[a],index=False)
            print(path_li[a]+'保存完成')

# str_li = ['辐照度^2','辐照度 风速',
#  '辐照度^3' ,'辐照度^2 风速' ,'辐照度 风速^2' ,'辐照度 风向^2', '风速^2 湿度', '辐照度 温度^2', '辐照度 湿度^2',
#  '辐照度 压强^2', '风速^2 风向', '风向^2 湿度', '温度^3', '湿度^3', '辐照度 风速^3', '辐照度 风速 风向^2',
#  '辐照度 风速 温度^2', '辐照度 风速 湿度^2', '辐照度 风速 压强^2', '辐照度^4' ,'辐照度^3 风速' ,'风向^3 湿度',
#  '风速 温度^3']
# stand_2_one_2(path_li,str_li)

def change_file(path):
    with open(path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        data['时间'] = [int(time.mktime(time.strptime(a, "%Y-%m-%d %H:%M:%S"))) for a in data['时间']]
        data['年'] = [int(time.strftime("%Y", time.localtime(i))) for i in data['时间']]
        data['月'] = [int(time.strftime("%m", time.localtime(i))) for i in data['时间']]
        data['日'] = [int(time.strftime("%d", time.localtime(i))) for i in data['时间']]
        data['时'] = [int(time.strftime("%H", time.localtime(i))) for i in data['时间']]
        data['分'] = [int(time.strftime("%M", time.localtime(i))) for i in data['时间']]
        data['时间'] = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(i))) for i in data['时间']]
        data['白天'] = data['辐照度'].apply(lambda x:1 if x != -1.0 else 0)
        data.to_csv(path,index=False)
        print(path+'保存完成')


def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

def scalars(path):
    with open(path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        data['年_1'] = [Z_ScoreNormalization(i, np.average(data['年']), np.std(data['年'])) for i in data['年']]
        data['月_1'] = [Z_ScoreNormalization(i,np.average(data['月']),np.std(data['月'])) for i in data['月']]
        data['日_1'] = [Z_ScoreNormalization(i, np.average(data['日']), np.std(data['日'])) for i in data['日']]
        data['时_1'] = [Z_ScoreNormalization(i, np.average(data['时']), np.std(data['时'])) for i in data['时']]
        data['分_1'] = [Z_ScoreNormalization(i, np.average(data['分']), np.std(data['分'])) for i in data['分']]
        data.to_csv(path,index=False)
        print(path+'保存完成')


# power_path = r'E:\国能竞赛\dataset\new_dataset\气象数据\电站1_气象.csv'
# train_path = r'E:\国能竞赛\dataset\new_dataset\train\train_1.csv'
# test_path = r'E:\国能竞赛\dataset\new_dataset\test\test_1.csv'

# power_li = []
# path_li = []
# for root, dirs, files in os.walk(r'E:\国能竞赛\dataset\new_dataset\气象数据'):
#     for file in files:
#         power_li.append(os.path.join(root,file))
# for root, dirs, files in os.walk(r'E:\国能竞赛\dataset\new_dataset\test'):
#     for file in files:
#         path_li.append(os.path.join(root,file))
def power_change(power_path,path):
    with open(power_path,'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        data['时间'] = [time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(i, "%Y-%m-%d %H:%M")) for i in data['时间']]
    with open(path,'r',encoding='utf-8') as f2:
        train_data = pd.read_csv(f2)
        new_data = pd.merge(data[['时间','总辐射','直辐射','散辐射']],train_data,how='right',on='时间')
        new_data.to_csv(path,index=False)
        print(path+'保存完成')
