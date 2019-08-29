import urllib.request
import os
import tarfile
import re
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,SimpleRNN,LSTM
from keras.layers.embeddings import Embedding

#数据下载
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="E:/work/pycharm_space/dataSet/imdb/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
#解压
if not os.path.exists("E:/work/pycharm_space/dataSet/imdb/aclImdb"): #如果文件没有解压，便对下载好的压缩包进行解压
    tfile = tarfile.open("E:/work/pycharm_space/dataSet/imdb/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('E:/work/pycharm_space/dataSet/imdb/')


# 利用正则将html中的标签去除
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

#acIImdb---train---------pos
#       |          |-----neg
#       |
#       |-test----------pos
#                  |-----neg


#创建函数用于读取设定格式的imdb信息
def read_files(filetype):
    path = "E:/work/pycharm_space/dataSet/imdb/aclImdb/"
    file_list = []
    positive_path = path + filetype + "/pos/"    #正面评价文件目录
    for f in os.listdir(positive_path):             #正面评价文件目录下所有文件添加到file_list中
        file_list += [positive_path + f]
    negative_path = path + filetype + "/neg/"    #负面评价文件目录
    for f in os.listdir(negative_path):             #负面评价文件目录下所有文件添加到file_list中
        file_list += [negative_path + f]
    print('read', filetype, 'files:', len(file_list))
    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_texts

#load样本
y_train_label,x_train_text = read_files("train")
y_test_label,x_test_text = read_files("test")
print('train text len:',len(x_train_text))
print('train label len:',len(y_train_label))
print('-------------------------------')

#查看第0项数据
print('train txet 0:')
print(x_train_text[0])
print('train label 0:')
print(y_train_label[0])


#建立token
#读取train一共25000个text样本,统计出现次数排名3800的单词,建立一个3800单词的词典
token = Tokenizer(num_words=3800)
token.fit_on_texts(x_train_text)
#token
print('token:')
print(token.word_index)

#英文单词转数字(txet文本转数字列表)
x_train_seq = token.texts_to_sequences(x_train_text)
x_test_seq = token.texts_to_sequences(x_test_text)
print('-----------------------------')
print('x_train_text[0]:') #第0个文本
print(x_train_text[0])
print('x_train_seq[0]:') #第0个文本转的数字列表
print(x_train_seq[0])
print('-----------------------------')

#txet文本数字列表统一长度为380
x_train = sequence.pad_sequences(x_train_seq,maxlen=380)
x_test = sequence.pad_sequences(x_test_seq,maxlen=380)

#字库大小为3800 则每个单词独热编码为3800维 每个样本有380个单词 则样本在word2ver之前为 3800*380(高阶稀疏矩阵)
#样本在word2ver之后每个单词变为32维 每个样本有380个单词 则样本变为 380*32(低阶稠密矩阵)

#拼接网络
model = Sequential()
model.add(Embedding(input_dim=3800,input_length=380,output_dim=32)) #100个单词每个单词转为32维稠密向量 即100*32的张量
model.add(Dropout(0.2)) #防止过拟合
#这里flatton层换为rnn层
model.add(LSTM(units=32)) #16个神经元
model.add(Dense(units=256,activation='relu')) #隐层宽度 256个神经元
model.add(Dropout(0.2)) #防止过拟合
model.add(Dense(units=1,activation='sigmoid')) #输出层1个神经元 0负面/1正面
print('model:',model.summary())

#设置超参数
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练
train_history = model.fit(x=x_train,y=y_train_label,validation_split=0.2,epochs=10,batch_size=100,verbose=2)



def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])          #train(训练准确率)这条线
    plt.plot(train_history.history[validation])     #validation(验证准确率)这条线
    plt.title('train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

#训练集准确率 误差率
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
#测试集准确率 误差率
scores = model.evaluate(x_test,y_test_label,verbose=0)
print('scores:',scores[1])

print('---------------------------------------------------------')

#预测
predict = model.predict_classes(x_test) #预测结果
predict_classes = predict.reshape(-1) #预测结果二维转一维
SentimentDict = {1:'正面',0:'负面'}
def display_test_Sentiment(i):
    print('index =',i)
    print(x_test_text[i])
    print('label:',SentimentDict[y_test_label[i]],'predict:',SentimentDict[predict_classes[i]])
#查看第2个测试样本的预测结果
display_test_Sentiment(2)

print('--------------------------------------------------')


#使用样本集之外的样本进行预测
#美女与野兽影评
input_text = '''
    Oh dear, oh dear, oh dear: where should I start folks. 
    I had low expectations already because I hated each and every single trailer so far, 
    but boy did Disney make a blunder here. 
    I'm sure the film will still make a billion dollars - hey: 
    if Transformers 11 can do it, 
    why not Belle? - but this film kills every subtle beautiful little thing that had made the original special, 
    and it does so already in the very early stages. 
    It's like the dinosaur stampede scene in Jackson's King Kong: only with even worse CGI (and, well, kitchen devices instead of dinos).
    The worst sin, though, is that everything (and I mean really EVERYTHING) looks fake. 
    What's the point of making a live-action version of a beloved cartoon if you make every prop look like a prop? 
    I know it's a fairy tale for kids, 
    but even Belle's village looks like it had only recently been put there by a subpar production designer trying to copy the images from the cartoon. 
    There is not a hint of authenticity here. 
    Unlike in Jungle Book, where we got great looking CGI, this really is the by-the-numbers version and corporate filmmaking at its worst. 
    Of course it's not really a bad film; those 200 million blockbusters rarely are (this isn't 'The Room' after all), 
    but it's so infuriatingly generic and dull - and it didn't have to be. In the hands of a great director the potential for this film would have been huge,
    Oh and one more thing: bad CGI wolves (who actually look even worse than the ones in Twilight) is one thing, 
    and the kids probably won't care. But making one of the two lead characters - Beast - look equally bad is simply unforgivably stupid. 
    No wonder Emma Watson seems to phone it in: she apparently had to act against an guy with a green-screen in the place where his face should have been
'''
input_seq = token.texts_to_sequences([input_text]) #单词转数字
pad_input_seq = sequence.pad_sequences(input_seq,maxlen=380) #保留380个数字
result = model.predict_classes(pad_input_seq) #预测
print('样本集外样本:')
print('result:',result,SentimentDict[result[0][0]])

