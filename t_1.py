'''
 * @desc : mlp训练mnist
 * @auth : TYF
 * @date : 2019/8/24 - 23:54
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#load数据(C://Users/kevin/.keras/datasets)
(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()
print('-------------------------------')
print('train data size:',len(x_train_image))
print('train label size:',len(y_train_label))
print('-------------------------------')
print('test data size:',len(x_test_image))
print('test label size:',len(y_test_label))
print('-------------------------------')
print('train data shape:',x_train_image.shape)
print('train label shape:',y_train_label.shape)

#show one pic
#def plot_image(image):
#    fig = plt.gcf()     #设置图像大小
#    fig.set_size_inches(2,2)    #设置图像大小
#    plt.imshow(image,cmap='binary')     #显示图像 binary为黑白灰图像
#    plt.show()  #绘图
#plot_image(x_train_image[0])

#show many pic
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()     #设置图像大小
    fig.set_size_inches(14,12)    #设置图像大小
    if num>25:num=25    #显示项数限定25
    for i in range(0,num):  #循环作图num个
        ax=plt.subplot(5,5,1+i) #子图为5行5列
        ax.imshow(images[idx],cmap='binary') #画子图
        title = "label="+str(labels[idx]) #标签作为title
        if len(prediction)>0: #如果预测结果不为空
            title += ",predict="+str(prediction[idx]) #预测结果拼接到title中
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([]) #隐藏刻度
        idx+=1
    plt.show()
#显示第0~第9张
#plot_image_labels_prediction(x_train_image,y_train_label,[],0,10)
print('-------------------------------')
#转1维向量
x_train = x_train_image.reshape(60000,784).astype('float32')
x_test = x_test_image.reshape(10000,784).astype('float32')
print('train data shape:',x_train.shape)
print('train label shape:',x_train.shape)
print('x_train[0]:',x_train[0])
#归一化
x_train_normalize = x_train /255;
x_test_normalize = x_test /255;
print('x_train_normalize[0]:',x_train_normalize[0])
print('-------------------------------')
#label转one-hot编码 60000train数据60000个label
y_train_label_onehot = np_utils.to_categorical(y_train_label)
y_test_label_onehot = np_utils.to_categorical(y_test_label)
print('y_train_label_onehot size:',len(y_train_label_onehot))


print('-------------------------------')
#输出层 : 10个label
model = Sequential()
#添加输入层/隐层
#隐层256个神经元
#输入层784神经元
#用normal distribution正态分布的随机数来初始化weight权重和bias偏差
#relu激活函数将
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
#添加输出层
#10个神经元
#用normal distribution正态分布的随机数来初始化weight权重和bias偏差
#softmax激活函数(将分类映射为概率)
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print('summary:')
print(model.summary())
#输入层->隐层 : h=relu(x*w1+b1)
#隐层->输出层 : y=softmax(h*w2+b2)
#每一层param计算方式 : param = (上一层神经元个数)x(本层神经元个数)+本层神经元个数
print('-------------------------------')
#categorical_crossentropy交叉熵作为损失函数
#adam优化器加速收敛
#accuracy(准确率)作为模型评分方式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#splite=2 4/5作为训练集 1/5作为验证集
#verbose=2 添加训练过程监听
#迭代5次 每次均进行模型评分
train_history = model.fit(x=x_train_normalize,y=y_train_label_onehot,validation_split=0.2,epochs=5,batch_size=200,verbose=2)
#Epoch 1/5
# - 3s - loss: 0.4387 - acc: 0.8831 - val_loss: 0.2197 - val_acc: 0.9382
#Epoch 2/5
# - 1s - loss: 0.1922 - acc: 0.9451 - val_loss: 0.1592 - val_acc: 0.9553
#Epoch 3/5
# - 1s - loss: 0.1352 - acc: 0.9609 - val_loss: 0.1256 - val_acc: 0.9639
#Epoch 4/5
# - 1s - loss: 0.1027 - acc: 0.9705 - val_loss: 0.1071 - val_acc: 0.9678
#Epoch 5/5
# - 1s - loss: 0.0813 - acc: 0.9768 - val_loss: 0.0970 - val_acc: 0.9721

#图像显示训练结果
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])          #train(训练准确率)这条线
    plt.plot(train_history.history[validation])     #validation(验证准确率)这条线
    plt.title('train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

#训练准确率、验证准确率对比图
show_train_history(train_history,'acc','val_acc')
#acc(训练准确率): 用训练数据来计算准确率,因为某些数据已经被训练过了又用来参与计算准确率所以会更高一点
#val_acc(验证准确率):  用验证集来计算准确率,未参与训练所以计算准确率会低一点
#训练误差、验证集误差对比图
show_train_history(train_history,'loss','val_loss')
#过拟合: 训练集准确率持续增加,验证集准确率并未增加。说明模型考虑了训练集样本中一些无关feature
print('-------------------------------')
#用测试集评估模型
scores = model.evaluate(x_test_normalize,y_test_label_onehot)
print('scores:')
print(scores[1])
print('-------------------------------')
#预测
prediction = model.predict_classes(x_test)
for i in range(1,len(y_test_label)):
   print('预测结果',prediction[i],'真实值:',y_test_label[i])
#混淆矩阵图
pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predict'])    #显示A识别为B的个数
df = pd.DataFrame({'label':y_test_label,'predict':prediction})  #显示A被识别为B的样本
print(df[(df.label==5)&(df.predict==3)]) #5被识别为3
#     label  predict
#340       5        3
#1003      5        3
#1393      5        3
#2035      5        3
#2810      5        3
#3902      5        3
#5937      5        3
#5972      5        3
#6598      5        3
#9482      5        3
#显示5被识别为3的样本
plot_image_labels_prediction(x_train_image,y_test_label,prediction,idx=340,num=1)



#修改隐藏层神经元个数为1000,添加两个隐层,复杂化网络结构会造成过拟合,加入DropOut防止
model_1000 = Sequential()
model_1000.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
model_1000.add(Dropout(0.5))
model_1000.add(Dense(units=1000,kernel_initializer='normal',activation='relu'))
model_1000.add(Dropout(0.5))
model_1000.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print('summary:')
print(model_1000.summary())






