#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from scipy.fftpack import fft,ifft


Lens=6000
path = "Bear_data/train.csv"


list = pd.read_csv(path)
list = np.array(list)[:Lens]
list_a=list[1][1:-1]
x=range(0,len(list_a))


# Get data:
ecg = list_a  # 生成心电信号
index = []
data = []
for i in range(len(ecg)-1):
    X = float(i)
    Y = float(ecg[i])
    index.append(X)
    data.append(Y)

# Create wavelet object and define parameters
w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
print("maximum level is " + str(maxlev))
threshold = 0.08  # Threshold for filtering

# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

plt.figure()
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

mintime = 0
maxtime = mintime + len(data) + 1

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime-1])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()

y=datarec[mintime:maxtime]
yy=fft(y)                     #快速傅里叶变换
yreal = yy.real               # 获取实数部分
yimag = yy.imag               # 获取虚数部分

yf=abs(fft(y))                # 取模
yf1=abs(fft(y))/((len(x)/2))           #归一化处理
yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

xf = np.arange(len(y))        # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))]  #取一半区间

# 以下滤波
maxy = max(yf1) / 5
cond = yf1 < maxy
yf1[cond] = 0
maxy = max(yf) / 5
cond = yf < maxy
yf[cond] = 0
maxy = max(yf2) / 5
cond = yf2 < maxy
yf2[cond] = 0
y=ifft(yf)

#原始波形
plt.subplot(221)
plt.plot(x,y)
plt.title('Original wave')
#混合波的FFT（双边频率范围）
plt.subplot(222)
plt.plot(xf,yf,'r') #显示原始信号的FFT模值
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
#混合波的FFT（归一化）
plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('FFT of Mixed wave',fontsize=10,color='#F08080')
plt.show()
