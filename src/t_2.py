'''
 * @desc : cnn训练mnist
 * @auth : TYF
 * @date : 2019/8/24 - 23:55
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential
np.random.seed(10)

#数据预处理
(x_train,y_train),(x_test,y_test) = mnist.load_data();
print('x_train.shape():',x_train.shape)
#张量   60000*28*28   转   60000*28*28*1
x_train_4d = x_train.reshape(x_train.shape[0],28,28,1).astype('float32');
x_test_4d = x_test.reshape(x_test.shape[0],28,28,1).astype('float32');
print('x_train_4d.shape():',x_train_4d.shape)
#标准化
x_train_4d_normalize = x_train_4d/255;
x_test_4d_normalize = x_test_4d/255;
#label转onehot编码
y_train_onehot = np_utils.to_categorical(y_train);
y_test_onehot = np_utils.to_categorical(y_test);

#拼接网络
model = Sequential()
# 16个5x5卷积核 relu激活函数
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
# 2x2的下采样窗口
model.add(MaxPooling2D(pool_size=(2,2)))
# 36个5x5卷积核 relu激活函数
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
# 2x2的下采样窗口
model.add(MaxPooling2D(pool_size=(2,2)))
# dropout层防止过拟合
model.add(Dropout(0.25))
# 平坦层压成1维 1764个神经元
model.add(Flatten())
# 隐层128个神经元
model.add(Dense(units=128,activation='relu'))
# dropout层防止过拟合
model.add(Dropout(0.5))
# 输出层softmax将类别映射为概率
model.add(Dense(units=10,activation='softmax'))
print(model.summary())

#训练
# categorical_crossentropy/交叉熵作为损失函数  adam优化器加速收敛  accuracy(准确率)作为模型评分方式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 4/5训练集 1/5验证集
train_history = model.fit(x=x_train_4d_normalize,y=y_train_onehot,validation_split=0.2,epochs=20,batch_size=300,verbose=2)


#图像显示训练结果
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])          #train(训练准确率)这条线
    plt.plot(train_history.history[validation])     #validation(验证准确率)这条线
    plt.title('train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

#训练集准确率/验证集准确率
show_train_history(train_history,'acc','val_acc')
#训练集误差/验证集误差
show_train_history(train_history,'loss','val_loss')
#测试集合准确率
scores = model.evaluate(x_test_4d_normalize,y_test_onehot)
print('scores:',scores[1])


#预测
prediction = model.predict_classes(x_test_4d_normalize)
for i in range(1,len(y_test)):
   print('predict:',prediction[i],'label:',y_test[i])


#混淆矩阵
print(pd.crosstab(y_test,prediction,rownames=['label'],colnames=['predict']))