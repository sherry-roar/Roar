#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "Bear_data/new_train.csv"
Lens=600



list = pd.read_csv(path)
list = np.array(list)[:6000]

for i in range(4):
    list_a=list[i+5][1:-1]
    X=range(0,len(list_a))
    plt.subplot(211)
    plt.plot(X,list_a)


    '''df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
    plt.figure(figsize=(10,4))'''
    # 创建图表、数据
    list_b=pd.Series(list_a)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    plt.subplot(212)
    f = list_b.plot.box(
                grid = True,
                color = color,
    )
    plt.title('boxplot')
    print(f)

    plt.grid(linestyle='--')
    plt.show()

'''plt.plot(X,C)
plt.plot(X,S)
#在ipython的交互环境中需要这句话才能显示出来
plt.show()'''