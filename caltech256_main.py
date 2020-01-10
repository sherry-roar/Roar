#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

train_PATH='./data.csv'

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import LabelEncoder


Batch_size = 80
Long = 20424
Lens = int(Long*0.8)

test_len=100

def load_img(filepath):
    #  从路径中读取图片
    images = []
    temp=Image.open(filepath)
    temp=temp.resize([224,224])
    temp=np.array(temp)
    img=temp.astype(np.float32)
    img=np.multiply(img,1.0/255.0)# 归一化
    if len(img.shape)!=3:
        # 处理单通道数据
        t=np.zeros((img.shape[0],img.shape[1],3))
        t[:,:,0]=img
        t[:, :, 1] = img
        t[:, :, 2] = img
        img=t
    images.append(img)
    images=np.transpose(images,(0,3,1,2))
    return images

def waitedata(x):
    # 数据白化，输入四维数据
    if x is not None:
        x[:, 0, :, :] = (x[:, 0, :, :] - 0.485) / 0.229
        x[:, 1, :, :] = (x[:, 1, :, :] - 0.456) / 0.224
        x[:, 2, :, :] = (x[:, 2, :, :] - 0.406) / 0.225
    x = np.transpose(x, (0, 2, 3, 1))
    return x

def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

def xs_gen(path,batch_size = Batch_size,train=True):
    img_path=path
    encoder = LabelEncoder()
    if train:
        img_list = np.array(img_path[:Lens])
        img_list[:, -1] = encoder.fit_transform(img_list[:, -1])
        np.random.shuffle(img_list)
        print("Found %s train items." % len(img_list))
        print("list 1 is", img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = np.array(img_path[Lens:])
        img_list[:, -1] = encoder.fit_transform(img_list[:, -1])
        np.random.shuffle(img_list)
        print("Found %s test items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):
            img=np.zeros((50,3,224,224))
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            path_x = np.array([file for file in batch_list[:,0]])
            i=0
            for path in path_x:
                t = load_img(path)
                img[i] = t[0]
                t += 1
            batch_x=waitedata(img)
            batch_y = np.array([convert2oneHot(label,210) for label in batch_list[:,-1]])

            yield batch_x, batch_y


def ts_gen(path,batch_size = Batch_size):
    # 若有用于检验我们模型的数据，可以输入
    img_list = x
    img_path = path
    print("Found %s test items."%len(img_list))
    print("list 1 is",img_list[0,-1])
    steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch

    while True:
        for i in range(steps):
            img=np.zeros((50,3,224,224))
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            path_x = np.array([file for file in batch_list[:,0]])
            i=0
            for path in path_x:
                t = load_img(path)
                img[i] = t[0]
                t += 1
            batch_x=waitedata(img)

            yield batch_x

shapedata=244*244

def build_model(input_shape=(224,224,3),num_classes=210):
    model = Sequential()
    # model.add(Reshape((shapedata, 1), input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu',padding="same",input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(64, (3,3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(128, (3,3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(1024))
    # model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)

def plot_training(history):
    acc = history.history['acc']
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r-')
    # plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r-')
    # plt.plot(epochs, val_loss, 'b-')
    plt.title('Training and validation loss')
    plt.show()


Train = True

if __name__ == "__main__":

    impath_list = pd.read_csv(train_PATH,header=None)
    if Train == True:
        train_iter = xs_gen(impath_list)
        val_iter = xs_gen(impath_list,train=False)
        ckpt = keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
            monitor='val_loss', save_best_only=True, verbose=1)

        model = build_model()
        opt = Adam(0.0002)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])
        print(model.summary())

        history=model.fit_generator(
            generator=train_iter,
            steps_per_epoch=Lens // Batch_size,
            epochs=10,
            initial_epoch=0,
            validation_data=val_iter,
            nb_val_samples=(Long - Lens) // Batch_size,
            callbacks=[ckpt],
        )
        model.save("finishModel.h5")
        plot_training(history)

        '''# plot curve
        print(history.history.keys())
        # acc
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc.jpg')
        # loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.jpg')'''

    else:
        test_iter = ts_gen()
        model = load_model("best_model.49-0.00.h5")
        pres = model.predict_generator(generator=test_iter,steps=math.ceil(test_len/Batch_size),verbose=1)
        print(pres.shape)
        ohpres = np.argmax(pres,axis=1)
        print(ohpres.shape)
        #img_list = pd.read_csv(TEST_MANIFEST_DIR)
        df = pd.DataFrame()
        df["id"] = np.arange(1,len(ohpres)+1)
        df["label"] = ohpres
        df.to_csv("final.csv",index=None)
        test_iter = ts_gen()
        for x in test_iter:
            x1 = x[0]
            break
        plt.plot(x1)
        plt.show()


