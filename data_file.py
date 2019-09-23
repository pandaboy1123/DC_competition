import pandas as pd
import numpy as np
# path = r'E:\国能竞赛\demo\sample_new.csv'
# with open(path,'r',encoding='utf-8') as f:
#     data = pd.read_csv(f)
#     number = len(data['prediction']<=0)
#     data.loc[data['prediction'] <= 0, 'prediction'] = 0
#     data.to_csv('sample1.csv',index=False)
#     print('修改完成，%s受到影响'%number)

path_1 = r'E:\国能竞赛\demo\change_test.csv'
path_2 = r'E:\国能竞赛\demo\cost.csv'
with open(path_1,'r',encoding='utf-8') as f1:
    data_1 = pd.read_csv(f1)
with open(path_2,'r',encoding='utf-8') as f2:
    data_2 = pd.read_csv(f2)
# print(data_1.head())
# print(data_2.head())
data_1['实际辐照度'] = data_2['real_p']
print(data_1.head())
data_1.to_csv('test_data.csv',index=False)