#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

from rpyc import Service
from rpyc.utils.server import ThreadedServer

class TestService(Service):


    def exposed_hello(self,ip):
        print(ip+' hello')
        return ip+'NB'

if __name__=="__main__":
    server = ThreadedServer(TestService, port=1235)
    server.start()