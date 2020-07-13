#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import socket

# 1.创建socket对象
s = socket.socket()

# 获取本地主机名
host = socket.gethostname()

# 设置端口
port = 12345

# 2.绑定端口
s.bind((host, port))

# 3.等待客户端连接，监听socket对象
s.listen(5)

while True:
    c, addr = s.accept()  # 建立客户端连接
    print('连接地址：', addr)
    # msg='欢迎访问百度！'
    # msg=msg.encode("UTF-8")
    # c.send(msg)
    # c.close()