'''
 * @desc : 多类别分类
 * @auth : TYF
 * @date : 2019/8/25 - 1:12
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D

np.random.seed(10)

#数据预处理
#加载数据(C:\Users\2019\.keras\datasets\cifar-10-batches-py.tar)
(x_img_train,y_label_train),(x_img_test,y_label_test) = cifar10.load_data();
print('x_img_train shape:',x_img_train.shape)   #(50000, 32, 32, 3)  50000张32x32的RGB图片
print('y_label_train shape:',y_label_train.shape)   #(50000, 1)
#标准化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
#标签转onehot编码
y_label_train_onehot = np_utils.to_categorical(y_label_train);
y_label_test_onehot = np_utils.to_categorical(y_label_test);
print('y_label_train shape:',y_label_train_onehot.shape)   #(50000, 10)



#图片显示
label_dict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()     #设置图像大小
    fig.set_size_inches(14,12)    #设置图像大小
    if num>25:num=25    #显示项数限定25
    for i in range(0,num):  #循环作图num个
        ax=plt.subplot(5,5,1+i) #子图为5行5列
        ax.imshow(images[idx],cmap='binary') #画子图
        title = str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0: #如果预测结果不为空
            title += '=>'+label_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([]) #隐藏刻度
        idx+=1
    plt.show()

plot_image_labels_prediction(x_img_train,y_label_train,[],0)


#拼接网络
modul = Sequential()
modul.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))  #padding='same'可以让原图经过filter后不改变大小
modul.add(Dropout(rate=0.25))
modul.add(MaxPooling2D(pool_size=(2,2)))
modul.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
modul.add(Dropout(rate=0.25))
modul.add(MaxPooling2D(pool_size=(2,2)))
modul.add(Flatten()) #这一层展平
modul.add(Dropout(rate=0.25))
modul.add(Dense(1024,activation='relu'))  #添加全连接层 1024神经元
modul.add(Dropout(rate=0.25))
modul.add(Dense(10,activation='softmax'))     #输出层  10神经元
print(modul.summary())

#接着上一次训练
try:
    modul.load_weights('E:/work/pycharm_space/keras/venv/saveModel/t_3_model.h5')
    print('加载成功!')
except:
    print('加载失败!')

modul.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = modul.fit(x_img_train_normalize,y_label_train_onehot,validation_split=0.2,epochs=5,batch_size=128,verbose=1)

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
#测试集误差
scores = modul.evaluate(x_img_test_normalize,y_label_test_onehot,verbose=0)
print('scores:',scores)

#预测
prediction = modul.predict_classes(x_img_test_normalize)
for i in range(1,len(y_label_test)):
   print('predict:',prediction[i],'label:',y_label_test[i][0],'name:',label_dict[y_label_test[i][0]])
#前10张预测结果
plot_image_labels_prediction(x_img_test,y_label_test,prediction,0,10)


#预测结果
predicted_probability = modul.predict(x_img_test_normalize)

#样本index是各个label的可能性
def show_probability(index):
    print('样本真实值:',label_dict[y_label_test[index][0]],'样本观测值:',label_dict[prediction[index]])
    for i in range(len(label_dict)):
        print('该样本为', label_dict[i] + ' 的可能性: %1.9f' % (predicted_probability[index][i]))

show_probability(0)


#显示混淆矩阵 predict=x label=y的预测结果样例数
print('混淆矩阵:')
print(
pd.crosstab(
    y_label_test.reshape(-1),   # 转成 1x10000 的
    prediction,                 #      1x10000 的
    rownames=['label'],
    colnames=['predict'] )
)

print('save model to disk !')
modul.save_weights('E:/work/pycharm_space/keras/venv/saveModel/t_3_model.h5')



#保存模型
#print('save model to disk !')
#modul.save_weights('saveModel/model.h5')
#modul.load_weights('saveModel/model.h5')
# epochs=100 可以设置epoche=10然后分十次训练 每次训练完调用save_weights保存 每次训练开始调用load_weights加载上次结果


#save()
#训练前调用: 保存网络
#训练后调用: 保存网络、权重
#save_weights
#训练后调用: 保存权重
