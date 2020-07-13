#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'


import os


# conn = rpyc.connect('192.168.43.90', 1235)
# task = conn.root.hello('myj')
# print(task)
if not os.path.exists('backup.pkl'):  # 创建一个文件夹
    f = open('backup.pkl', 'w')
    f.close()

    exposed_get_files(self)

    exposed_read(self,fname)

    exposed_write(self,dest,size)

    exposed_back_ups(self,dest,size)

    heartbeat(self)

