import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
np.random.seed(10)

#导入数据集，如果没有就会自动下载
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
print("train:",len(x_img_train))
print("test:",len(x_img_test))
print("train_image:",x_img_train.shape)
print("train_label:",y_label_train.shape)
print("test_label:",y_label_test.shape)

#3.查看图片像素
print(x_img_test[0])

#4.可视化部分训练集
label_dict ={0:"airplane",1:"automobile",2:"bird",3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

#可视化25张待预测图片
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    if num>25: num = 25 #最多显示25张图片
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title = str(i)+","+label_dict[labels[i][0]] #i-th张图片对应的类型
        if len(prediction) >0:
            title +='=>'+label_dict[prediction[i]]
        ax.set_title(title,fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.savefig("1.pang")
    plt.show()

#将类别归一化
print(x_img_train[0][0][0])  #显示图片数量及像素
x_img_train_normalize = x_img_train.astype("float32")/255.0
x_img_test_normalize = x_img_test.astypr('float32')/255.0
print(x_img_train_normalize[0][0][0])
print(y_label_train.shape)
print(y_label_test.shape)

from keras.utils import np_utils
y_label_train_Onehot = np_utils.to_categorical(y_label_train)
y_label_test_Onehot = np_utils.to_categorical(y_label_test)
print(y_label_train_Onehot.shape)
print(y_label_train_Onehot[:5])

#设置卷积神经网络的参数
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

#设置模型参数
model = Sequential()
#模型：卷积层1，池化层1
model.add(Conv2D(filter=32,keras_size=(3,3),input_shape=(32,32,3),activation='relu',
                 padding = 'padding'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))  #此层图片大小16*16

#模型构建：卷积层2，池化层2
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 activation='relu',padding='same'))
model.add(Dropout(0.25))
model.add(Dense(1024,activation='relu'))  #FC2 10224
model.add(Dropout(rate=0.25))
model.add(Dense(10,activation='softmax'))  #output 10
print(model.summary())

import os
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_histroy = model.fit(x_img_train_normalize,y_label_train_Onehot,validation_split=0.2,
                          epochs=2,batch_size=128,verbose=1)
#verbose: 日志显示，0为不输出日志信息，1为输出进度条纪录，2为每个epoch输出一行纪录。

#可视化训练过程
import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_histroy.history[train_acc])
    plt.plot(train_histroy.history[test_acc])
    plt.title("train history")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(['train'.'test'],loc = 'upper left')
    plt.savefig('1.png')
    plt.show()

#可视化
show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

#模型评估
print(model.metrics_names)
scorces = model.evaluate(x_img_test_normalize,
                         y_label_test_Onehot,verbose=1)
print(scorces)
print(scorces[1])

#预测结果
prediction = model.predict_classes(x_img_test_normalize)
prediction[:10]


#可视化部分预测结果
def plot_images_labels_predictions(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25: num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(images[idx],cmap='binary')
        title =str(i) + ',' +lable_dict[labels[i][0]]
        if len(prediction)>0:
            title+="=>"+label_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

#显示预测结果
plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,10)

#查看预测概率
def show_predicted_probability(y,prediction,x_img,predicted_probability,i):
    print("label:",label_dict[y[i][0]],"predict:",label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_img_test[i],(32,32,3)))
    plt.show()


#显示混淆矩阵
print(y_label_test.reshape(-1)) #转化为1维数组

try:model.load_weights("cifarCnnModel.h5"):\
    print("载入模型成功，继续训练模型")
except:
    print("载入模型失败，卡是训练新模型")


#开始训练
model.compile(loss="categorical_crossentropy",
              optimizer='adam',metrics=['accuracy'])

train_histroy=model.fit(x_img_train_normalize,y_label_train_Onehot,
                        validation_split=0.2,epochs=3,batch_size=128,verbose=1)

import os
os.mkdir("SaveModel")
model.sample_weights("SaveModel/cifar10CnnModel.h5")
print("saved model to disk")

