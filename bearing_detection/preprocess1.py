#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from scipy.fftpack import fft,ifft



Lens=6000
path = "Bear_data/test_data.csv"
w_path = "Bear_data/new_test.csv"


list = pd.read_csv(path)
list = np.array(list)[:]
Lens=len(list[1])-1
datawt=np.zeros((528,Lens))

# for i in range(len(list)):
for i in range(5):
    print('this is the',i,'round')
    ecg=list[i][1:]
    x=range(0,len(ecg))
    # 1、小波平滑部分，使用db8小波，阈值0.04
    index = []
    data = []
    for j in range(len(ecg)):
        X = float(j)
        Y = float(ecg[j])
        index.append(X)
        data.append(Y)
    w=pywt.Wavelet('db8')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # print("maximum level is " + str(maxlev))
    threshold = 0.04  # Threshold for filtering
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    # plt.figure()
    for j in range(1, len(coeffs)):
        coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))
        # 将噪声滤波
    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

    mintime = 0
    maxtime = mintime + len(data) + 1
    # plt.plot(index[mintime:maxtime], datarec[mintime:maxtime - 1])
    # plt.xlabel('time (s)')
    # plt.ylabel('microvolts (uV)')
    # plt.title("De-noised signal using wavelet techniques")
    # plt.tight_layout()
    # plt.show()

    xwt=index[mintime:maxtime]
    ywt=datarec[mintime:maxtime - 1]

    # 2、FFT 使用0.2阈值去噪
    yy = fft(ywt)  # 快速傅里叶变换
    yreal = yy.real  # 获取实数部分
    yimag = yy.imag  # 获取虚数部分

    yf = abs(fft(ywt))  # 取模
    yf1 = abs(fft(ywt)) / ((len(xwt) / 2))  # yf混合波的FFT（归一化）
    xf = np.arange(len(ywt))  # 频率

    # 以下滤波
    maxy = max(yf1) / 5
    cond = yf1<maxy
    yf1[cond]=0

    # 逆变换还原波形
    yfn = abs(ifft(yf1))
    datawt[i]=yfn
    plt.plot(x, 100*yfn, 'b')
    plt.show()
    # datawt.append(yfn)
    # yfn=np.transpose(yfn)


np.savetxt(w_path, datawt, delimiter = ',')
# df1 = pd.DataFrame({'A': datawt})
# df1.to_csv(w_path,header=None,index=None,mode = 'a')



