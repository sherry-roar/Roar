#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import os
# import matplotlib.pyplot as plt
import pandas as pd

# ospath = os.getcwd()    #获取当前路径
path="./data/256_ObjectCategories"

# x=[]
# y=[]
# cc=0
# sums=0
for file in os.listdir(path):  # file 表示的是文件名
    # count = count + 1
    newpath=os.path.join(path,file)
    # y.append(file[4:])
    # print(newpath)
    for root,dirs,files in os.walk(newpath):    #遍历统计
        count = 0
        for each in files:
            count += 1   #统计文件夹下文件个数
        # x.append(count)
        if count<=120 and count>=80 :
            # cc+=1
            # sums=sums+count
            for each in files:
                if each.endswith(".jpg"):
                    # print (file[4:],os.path.join(newpath,each))
                    my_list=[os.path.join(newpath,each),file[4:]]
                    df = pd.DataFrame(data=[my_list])
                    df.to_csv("./data.csv", encoding="utf-8-sig", mode="a", header=False, index=False)

# print(cc,sums)





'''plt.bar(range(len(x)),x)
# plt.plot(range(len(x)), x)
# plt.xticks(range(len(y)),y)
plt.ylabel("number")
plt.title("class")
plt.show()'''
