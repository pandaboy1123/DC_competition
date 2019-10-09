# -*- encoding: utf-8 -*-
"""
@File    : baseline.py
@Time    : 2019/10/8 16:06
@Author  : pandaboy
@Email   : pandaboy11223@gmail.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import Imputer
import seaborn as sns
import itertools
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cbt
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


target_path = r'E:\厦门银行数据\train_target.csv'
train_path = r'E:\厦门银行数据\train.csv'
test_path = r'E:\厦门银行数据\test.csv'
# col_name_l = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'edu', 'job', 'lmt', 'basicLevel','unpayIndvLoan']

with open(target_path,'r',encoding='utf-8') as f:
    target_data = pd.read_csv(f)
with open(train_path,'r',encoding='utf-8') as f:
    train_data = pd.read_csv(f)
with open(test_path,'r',encoding='utf-8') as f:
    test_data = pd.read_csv(f)
data = pd.merge(train_data,target_data,how='left',on='id')
col_names = data.columns.values.tolist()
col_names_test = test_data.columns.values.tolist()
col_names.remove('id')
col_names.remove('target')
col_names.remove('bankCard')
for i in col_names:
    data[i].fillna(int(data[i].mean()))
    median = data.loc[data[i] != -999, i].median()
    data[i] = np.where(data[i] ==-999, median,data[i])
col_names_test.remove('id')
for i in col_names_test:
    test_data[i].fillna(int(test_data[i].mean()))
    median = test_data.loc[test_data[i] != -999, i].median()
    test_data[i] = np.where(test_data[i] == -999, median,test_data[i])
print(col_names)
# abs(data.corr()['target']).to_csv('s.csv')
name = ['x_45','unpayOtherLoan', 'x_20',]
train_x = data[name]
train_y = data['target'].values
test = test_data[name]

# params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
#           'min_data_in_leaf': 30,
#           'objective': 'binary', #定义的目标函数
#           'max_depth': -1,
#           'learning_rate': 0.01,
#           "min_sum_hessian_in_leaf": 6,
#           "boosting": "gbdt",
#           "feature_fraction": 0.9,  #提取的特征比率
#           "bagging_freq": 1,
#           "bagging_fraction": 0.8,
#           "bagging_seed": 11,
#           "lambda_l1": 0.1,             #l1正则
#           'lambda_l2': 0.001,     #l2正则
#           "verbosity": -1,
#           "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
#           'metric': {'binary_logloss', 'auc'},  ##评价函数选择
#           # "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
#           # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
#           }
#
# folds = KFold(n_splits=5, shuffle=True)
# prob_oof = np.zeros((train_x.shape[0], ))
# test_pred_prob = np.zeros((test.shape[0], ))
#
#
# ## train and predict
# feature_importance_df = pd.DataFrame()
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(data)):
#     print("fold {}".format(fold_ + 1))
#     trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y[trn_idx])
#     val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])
#
#
#     clf = lgb.train(params,
#                     trn_data,
#                     1000,
#                     valid_sets=[trn_data, val_data],
#                     verbose_eval=20,
#                     early_stopping_rounds=60)
#     prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)
#
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["Feature"] = col_names
#     fold_importance_df["importance"] = clf.feature_importance()
#     fold_importance_df["fold"] = fold_ + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#     prob_oof += clf.predict(data[col_names_test], num_iteration=clf.best_iteration) / folds.n_splits
#     test_pred_prob += clf.predict(test_data[col_names_test], num_iteration=clf.best_iteration) / folds.n_splits
# threshold = 0.008
# for i in range(len(prob_oof)):
#     if prob_oof[i]>threshold:
#         prob_oof[i] = 1
#     else:
#         prob_oof[i] = 0
# print(len([i for i in prob_oof if i==1]))
# print(roc_auc_score(data['target'],prob_oof))
# for i in range(len(test_pred_prob)):
#     if test_pred_prob[i]>threshold:
#         test_pred_prob[i] = 1
#     else:
#         test_pred_prob[i] = 0

# class_names = np.array(['0','1'])
# cm = confusion_matrix(data['target'],prob_oof)
# plot_confusion_matrix(cm,class_names)
X_train,x_test,Y_train,y_test = train_test_split(train_x,train_y,test_size=0.4)
# iterations=200, depth=12, learning_rate=0.01, loss_function='Logloss',custom_metric='AUC',
#                               logging_level='Verbose',nan_mode='Min'
R_tree = cbt.CatBoostClassifier()
# dicts = {
#     "max_depth": [1,3,15,25],
#     "n_estimators": [2,5,13,20,35]
# }
# R_tree = GridSearchCV(R_tree,param_grid=dicts)
R_tree.fit(X_train, Y_train)
y_ = R_tree.predict(x_test)
class_names = np.array(['0','1'])
cm = confusion_matrix(y_test,y_)
plot_confusion_matrix(cm,class_names)
# print(R_tree.best_params_) #最佳参数
# print(R_tree.best_score_)  #最佳结果
print(roc_auc_score(y_test,y_))