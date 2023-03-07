#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pandas as pd
import math


path = "Bear_data/test1.csv"


#采样点选择1400个，因为设置的信号频率分量最高为600Hz，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400Hz（即一秒内有1400个采样点）
# x=np.linspace(0,1,1400)

#设置需要采样的信号，频率分量有180，390和600
# y=7*np.sin(2*np.pi*180*x) + 1.5*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

# Lens=6000



list = pd.read_csv(path,header=None)
list = np.array(list)

x=range(6000)
xf=range(3000)
# sum=03000
for i in range(len(list)):
    y=list[i][:-1]
    yy=fft(y)                     #快速傅里叶变换

    yf=abs(fft(y))                # 取模
    yf1=abs(fft(y))/((len(x)/2))           #归一化处理
    yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

    plt.subplot(221)
    plt.plot(x,y,'b')
    plt.title('Original wave',color='b')
    plt.subplot(222)
    plt.plot(xf,yf2,'r')
    plt.title('FFT wave',color='r')
    plt.show()
#     lens=len(list_a)
#     x=range(0,lens)
#     y=sorted(list_a)
#
#     q1=math.ceil((lens+1)/4)
#     q3=3*q1
#     nq1=y[q1]
#     nq3=y[q3]
#     iqr=nq3-nq1
#     up=nq3+1.5*iqr
#     down=nq1-1.5*iqr
#
#     a=np.where(y>up)
#     a=len(a[0])
#     b=np.where(y<down)
#     b=len(b[0])
#     sum=sum+a+b
# print(sum/(len(list)*6000))

# yy=fft(y)                     #快速傅里叶变换
# yreal = yy.real               # 获取实数部分
# yimag = yy.imag               # 获取虚数部分
#
# yf=abs(fft(y))                # 取模
# yf1=abs(fft(y))/((len(x)/2))           #归一化处理
# yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间
#
# maxy=max(yf1)/5
# for i in range(len(yf1)):
#     if yf[i]<maxy:
#         yf[i]=0
# yif1=ifft(yf)
#
# xf = np.arange(len(y))        # 频率
# xf1 = xf
# xf2 = xf[range(int(len(x)/2))]  #取一半区间
#
# #原始波形
# plt.subplot(221)
# plt.plot(x,y)
# plt.title('Original wave')
# #混合波的FFT（双边频率范围）
# plt.subplot(222)
# plt.plot(xf,yf,'r') #显示原始信号的FFT模值
# plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
# #混合波的FFT（归一化）
# plt.subplot(223)
# plt.plot(xf1,yif1,'g')
# plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
#
# plt.subplot(224)
# plt.plot(xf2,yf2,'b')
# plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
# plt.show()
#
#
#
# # plt.subplot(121)
# # plt.plot(xf1[20:-20],yif1[20:-20],'b')
# # plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
# # plt.subplot(122)
# list_b=pd.Series(yif1)
# color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
# f = list_b.plot.box(
#     grid=True,
#     color=color,
# )
# plt.show()
