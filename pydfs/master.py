import rpyc
import uuid
import threading 
import math
import random
import configparser
import signal
import sys
import os
import pickle  # 用来存储字典的那个工具

from rpyc.utils.server import ThreadedServer

def int_handler(signal, frame):
  pickle.dump((MasterService.exposed_Master.file_table,MasterService.exposed_Master.block_mapping),open('fs.img','wb'))
  sys.exit(0)

def set_conf():
  conf=configparser.ConfigParser()
  conf.read_file(open('dfs.conf'))
  MasterService.exposed_Master.block_size = int(conf.get('master','block_size'))
  MasterService.exposed_Master.replication_factor = int(conf.get('master','replication_factor'))
  minions = conf.get('master','minions').split(',')
  for m in minions:
    id,host,port=m.split(":")
    MasterService.exposed_Master.minions[id]=(host,port)

  # if os.path.isfile('fs.img'):
  #   MasterService.exposed_Master.file_table,MasterService.exposed_Master.block_mapping = pickle.load(open('fs.img','rb'))

def load_obj(name):  # 将存储的字典取出来
    try:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except EOFError:  # 捕获异常EOFError 后返回None
        return None


def save_obj(self, name):  # 用于存储字典
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

class MasterService(rpyc.Service):
  class exposed_Master():
    if not os.path.exists('backup.pkl'):
        file_table = {}
    elif os.path.getsize('backup.pkl')==0:
        file_table={}
    else:
        # file_table = load_obj('backup')
        file_table = {}

    # block_mapping = {}
    backup_minions={}
    minions = {}

    block_size = 0
    replication_factor = 0


    def __init__(self):
      self.heartbeat()
      lens=len(self.minions)
      if lens>1:
          backup_num=math.floor(lens)
          self.backup_minions=self.minions
          # print('backup minions is: ',self.backup_minions.get('2'),'\n')

    def exposed_get_files(self):
      dic = self.__class__.file_table
      if dic is not None:
          f=[]
          for k in dic:
              f.append(k)
          return f
      else:
          return None


    def exposed_read(self,fname):
      self.heartbeat()  # check connection
      mapping = self.__class__.file_table[fname]
      return mapping

    def exposed_write(self,dest,size):
      self.heartbeat()  # check connection
      if self.exists(dest):
        pass # ignoring for now, will delete it later

      self.__class__.file_table[dest]=[]

      num_blocks = self.calc_num_blocks(size)
      blocks = self.alloc_blocks(dest,num_blocks)
      flag=0
      if self.backup_minions is not None:
          flag=1

      return blocks,flag

    def exposed_back_ups(self,dest,size):
        blocks = []
        num = self.calc_num_blocks(size)
        for i in range(0, num):
            block_uuid = uuid.uuid1().hex
            nodes_ids = random.sample(self.__class__.minions.keys(),
                                      self.__class__.replication_factor)
            # replication_factor分块几次意思
            blocks.append((block_uuid, nodes_ids))

            self.__class__.file_table[dest].append((block_uuid, nodes_ids))
        return blocks

    def exposed_get_file_table_entry(self,fname):
      if fname in self.__class__.file_table:
        return self.__class__.file_table[fname]
      else:
        return None

    def exposed_get_block_size(self):
      return self.__class__.block_size

    def exposed_get_minions(self):
      return self.__class__.minions

    def exposed_get_backup_minions(self):
      return self.__class__.backup_minions

    def calc_num_blocks(self,size):
      return int(math.ceil(float(size)/self.__class__.block_size))

    def exists(self,file):
      return file in self.__class__.file_table

    def alloc_blocks(self,dest,num):
      blocks = []
      for i in range(0,num):
        block_uuid = uuid.uuid1().hex
        nodes_ids = random.sample(self.__class__.minions.keys(),self.__class__.replication_factor)
        # replication_factor分块几次意思
        blocks.append((block_uuid,nodes_ids))

        self.__class__.file_table[dest].append((block_uuid,nodes_ids))
        # save_obj(self.__class__.file_table, 'backup')

      return blocks

    def heartbeat(self):
        self.__class__.minions={}
        set_conf()
        # 检测连接，每次修改minion连接需要手动改dfs.conf
        m = self.__class__.minions
        print('minions is : ',m, '\n')
        return m

'''读写锁'''
class RWLock(object):
  def __init__(self):
    self.rlock = threading.Lock()
    self.wlock = threading.Lock()

    self.reader = 0

  def write_acquire(self):
      self.wlock.acquire()
  def write_release(self):
    self.wlock.release()

  def read_acquire(self):
      self.rlock.acquire()
      self.reader += 1
      if self.reader == 1:
        self.wlock.acquire()
      self.rlock.release()

  def read_release(self):
    self.rlock.acquire()
    self.reader -= 1
    if self.reader == 0:
      self.wlock.release()
    self.rlock.release()

if __name__ == "__main__":
  set_conf()
  if not os.path.exists('backup.pkl'):  # 创建一个文件夹
      f = open('backup.pkl', 'w')
      f.close()
  else:
      pass
  signal.signal(signal.SIGINT,int_handler)# 宕机（ctrl+c）后将操作存入fs.img
  t = ThreadedServer(MasterService, port = 12346)# 等待接受信号
  t.start()
