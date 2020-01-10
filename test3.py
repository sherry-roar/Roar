#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

train_PATH='./data.csv'

from PIL import Image
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()




batch_size = 20
Long = 20425
Lens = int(Long*0.8)

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

def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

img_path= pd.read_csv(train_PATH,header=None)


img_list = np.array(img_path[:Lens])
img_list[:,-1] = encoder.fit_transform(img_list[:,-1])

np.random.shuffle(img_list)
print("Found %s train items."%len(img_list))
print("list 1 is",img_list[0,1])
steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch


for i in range(steps):
    img=np.zeros((20,3,224,224))
    batch_list = img_list[i * batch_size : i * batch_size + batch_size]

    np.random.shuffle(batch_list)

    path_x = np.array([file for file in batch_list[:,0]])
    i=0
    for path in path_x:
        t=load_img(path)
        img[i]=t[0]
        t+=1
    batch_x=np.transpose(img,(0,2,3,1))

    batch_y = np.array([convert2oneHot(label,210) for label in batch_list[:,-1]])

    print(batch_y)