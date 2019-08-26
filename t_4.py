'''
 * @desc : 泰坦尼克号乘客生存率预测
 * @auth : TYF
 * @date : 2019/8/25 - 22:45
'''
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import os
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D

#load数据
url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
filePath = 'E:/work/pycharm_space/dataSet/titanic3/titanic3.xls'

if not os.path.isfile(filePath):
    result = urllib.request.urlretrieve(url,filePath)
    print('download result:',result)

all_df = pd.read_excel(filePath)

#查看1项数据
print(all_df[:1])
#从all_df筛选需要的字段到DataFrame中
#survived是label、其他均是特征字段
clos = ['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
all_df = all_df[clos]
print(all_df[:1])

# 数据预处理 返回样本和类别
def preDo(raw_df):
    # name      姓名字段在训练时不需要，必须先删除，但在预测阶段会使用
    # age     有些项的age字段是null，必须将null改为平均值
    # fare   同age
    # sex   性别字段是文字，需转换为0和1
    # embarked  登船港口有三个分类 需使用One-Hot Encoding 转换 相当于将一个特征转为三个特征
    # 删除name字段
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()  # 计算age平均值
    df['age'] = df['age'].fillna(age_mean)  # age null值替换为平均值
    fare_mean = df['fare'].mean()  # 计算fare平均值
    df['fare'] = df['fare'].fillna(age_mean)  # fare null值替换为平均值
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)  # 性别字符串转数字
    x_onehot_df = pd.get_dummies(data=df, columns=['embarked'])  # 仓位embarked转onehot编码 1字段转3字段
    print('x_onehot_df:', x_onehot_df[:2])
    # DataFrame转Array
    ndarray = x_onehot_df.values
    print('ndarray.shape:', ndarray.shape)  # (1309, 10) 1309个样本每个样本10个字段  字段0是label(survived)
    # 取特征字段 0列
    Label = ndarray[:, 0]
    # 取label字段 1-9列
    Feature = ndarray[:, 1:]
    # 样本各feature有价格年级等，先标准化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 标准刻度0-1
    scaledFeature = minmax_scale.fit_transform(Feature)  # 得到最终的用样本
    print('scaledFeature shape:', scaledFeature.shape)  # (1309, 9)
    return scaledFeature,Label

#2:8划分训练集和测试集
msk = np.random.rand(len(all_df))<0.8
train_df = all_df[msk]
test_df = all_df[~msk]

#分别预处理
train_feature,train_label = preDo(train_df)
test_feature,test_label = preDo(test_df)
print('train_feature:',train_feature.shape)
print('train_label:',train_label)
print('test_feature:',test_feature.shape)
print('test_label:',test_label)

