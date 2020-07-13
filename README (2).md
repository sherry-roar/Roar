### The (simplified) code presented at SRECon is located under [srecon](https://github.com/sanketplus/PyDFS/tree/srecon) branch.

---
# PyDFS
Simple (~200 lines) distributed file system like HDFS (and of-course GFS). It consists of one Master (NameNode) and multiple Minions (DataNode). And a client for interation. It will dump metadata/namespace when given SIGINT and reload it when fired up next time. Replicate data  the way HDFS does. It will send data to one minion and that minion will send it to next one and so on. Reading done in similar manner. Will contact first minion for block, if fails then second and so on.  Uses RPyC for RPC.

#### [Blog: Simple Distributed File System in Python : PyDFS](https://superuser.blog/distributed-file-system-python/) 

### Requirements:
  - rpyc (Really! That's it.)
  
### How to run.
  1. Edit `dfs.conf` for setting block size, replication factor and list minions (`minionid:host:port`)
  2. Fire up master and minions.
  3. To store and retrieve a file:
```sh
$ python client.py put sourcefile.txt sometxt
$ python client.py get sometxt
```
##### Stop it using Ctll + C so that it will dump the namespace.

## TODO:
  1. Implement Delete
  2. Use better algo for minion selection to put a block (currently random)
  3. Dump namespace periodically (maybe)
  4. Minion heartbeats / Block reports
  5. Add entry in namespace only after write succeeds.
  6. Use proper datastructure(tree-like eg. treedict) to store
     namespace(currently simple dict)
  7. Logging
  8. Expand this TODO
