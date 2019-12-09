#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import pandas as pd
import numpy as np
from scipy.fftpack import fft,ifft


path = "Bear_data/train.csv"
list = pd.read_csv(path)
list = np.array(list)[:,1:]
a=[]
long=len(list[1]-2)
for i in range(len(list)):
    if list[i,-1]==0:
        y=list[i,:-1]

        yy = fft(y)  # 快速傅里叶变换
        yreal = yy.real  # 获取实数部分
        yimag = yy.imag  # 获取虚数部分

        yf = abs(fft(y))  # 取模
        yf1 = abs(fft(y)) / ((long/ 2))  # yf混合波的FFT（归一化）
        yf2 = yf1[range(int(long / 2))]  # 由于对称性，只取一半区间

        # xf = np.arange(len(y))  # 频率
        # xf2 = xf[range(int(long / 2))]
        a.append(yf2)
print(a)
