import os
from typing import Set
import cv2
import sqlite3
from abc import ABC
import numpy as np
from collections import OrderedDict
import os
from multiprocessing import Process, Queue
import multiprocessing as mp
import queue
import turbojpeg
import time

class Cacher(ABC):
    #Input and outputs are in shape(512, 512, 4), however due to turbojpeg usage, the alpha channel might be inaccurate
    def read(hs:int) -> np.ndarray:
        pass
    def write(hs:int, data:np.ndarray):
        pass
    

class RAMCacher(Cacher):
    def __init__(self, max_size:int = 2, cache_quality:int = 90,  image_size:int = 512): #Size in GBs
        self.max_kbytes = max_size * 1024 * 1024
        self.cached_kbytes = 0
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
        self.cache_quality = cache_quality
        self.size = image_size
    def read(self, hs:int) -> np.ndarray:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            if self.cache_quality == 100:
                img = cached
            else:
                res = turbojpeg.decompress(cached, fastdct = True, fastupsample=True, pixelformat=turbojpeg.BGRA)
                img = np.ndarray((self.size,self.size,4), dtype=np.uint8, buffer=res)
            return img
        else:
            self.miss += 1
            return None
    def write(self, hs:int, data:np.ndarray):
        if self.cache_quality == 100:
            self.cache[hs] = np.array(data)
            self.cached_kbytes += data.nbytes /1024
            while self.cached_kbytes > self.max_kbytes:
                poped = self.cache.popitem(last=False)
                self.cached_kbytes -= poped[1].nbytes/1024
        else:
            compressed = turbojpeg.compress(data, self.cache_quality, turbojpeg.SAMP.Y420,fastdct = True, optimize= True, pixelformat=turbojpeg.BGRA)
            self.cache[hs] = compressed
            self.cached_kbytes += len(compressed) /1024
            while self.cached_kbytes > self.max_kbytes:
                poped = self.cache.popitem(last=False)
                self.cached_kbytes -= len(poped[1])/1024



class ReaderProcess(Process): # A seperate process that reads input from database, run in another process to avoid potential block in sqlite operation
    def __init__(self, read_trigger:Queue, 
                 read_return:Queue, 
                 db_path:str, max_size:int, #DB max ram size
                 ):
        super(ReaderProcess, self).__init__()
        self.read_trigger = read_trigger
        self.read_return = read_return
        self.db_path = db_path
        self.max_size = max_size

    def run(self):
        self.conn = self.create_db_for_read(self.db_path, self.max_size)
        self.read_return.put_nowait('ready')
        print('Reader start running')
        while True:
            try:
                hs = self.read_trigger.get(block=True, timeout=20.0)
            except queue.Empty:
                print('Reader trigger timeout or other exception')
                hs = None

            if hs is None:
                print('Reader gets ending signal')
                break
            ret = self.conn.execute('SELECT * FROM cache WHERE hash=?', (hs,)).fetchone()
            if ret is None:
                self.read_return.put_nowait((hs, None))
            else:
                self.read_return.put_nowait((hs, ret[1]))
        self.conn.close()
        print('Reader closed db')
        exit()

    def create_db_for_read(self, db_path:str, max_size:int):
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode = wal;')
        conn.execute('PRAGMA synchronous = normal;')
        conn.execute('PRAGMA temp_store = memory;')
        conn.execute('PRAGMA page_size = 4096;') # 4Kb Windows optimization
        conn.execute(f'PRAGMA mmap_size = {str(1024 * 1024 * 1024 * max_size/2)};')
        conn.execute(f'PRAGMA cache_size = -{str(1024 * 1024 * max_size/2)};')
        # Create the table
        conn.execute('CREATE TABLE IF NOT EXISTS cache( hash INTEGER PRIMARY KEY, bytes BLOB NOT NULL);')
        print('Existing entries:',conn.execute('SELECT count(*) from cache').fetchone()[0])
        return conn
                

class WriterProcess(Process): # A seperate process that write input to database, run in another process to avoid potential block in sqlite operation
    def __init__(self, write_queue:Queue, db_path:str, cache_quality:int = 90):
        super(WriterProcess, self).__init__()
        self.write_queue = write_queue
        self.db_path = db_path
        self.cache_quality = cache_quality

    def run(self):
        assert(os.path.isfile(self.db_path)) # Must already exist
        self.conn = self.db_for_write(self.db_path)
        print('Writer start runing')
        while True:
            try:
                res = self.write_queue.get(block=True, timeout=20.0)
            except queue.Empty:
                print('Writer trigger timeout or other exception')
                res = None
            if res is None:
                print('Writer get ending signal')
                break
            hs = res[0]
            if self.cache_quality != 100:
                cache_bytes = turbojpeg.compress(res[1], self.cache_quality, turbojpeg.SAMP.Y420,fastdct = True, optimize=True, pixelformat=turbojpeg.BGRA)
            else:
                cache_bytes = res[1].tobytes()
            self.conn.execute('INSERT OR IGNORE INTO cache(hash, bytes) VALUES (?, ?)',(hs, cache_bytes))
            self.conn.commit()
        self.conn.close()
        print('Writer closed db')
        exit()

    def db_for_write(self, db_path:str):
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode = wal;')
        conn.execute('PRAGMA synchronous = normal;')
        conn.execute('PRAGMA temp_store = memory;')
        # conn.execute(f'PRAGMA mmap_size = {str(1024 * 1024 * 1024 * max_size/2)};')
        # conn.execute(f'PRAGMA cache_size = -{str(1024 * 1024 * max_size/2)};')
        return conn



class DBCacherMP(Cacher):
    def __init__(self, max_size:int = 1,  db_dir:str = './cacher.sqlite', cache_quality:int = 90, image_size:int = 512) -> None:
        self.read_trigger = Queue()
        self.read_return = Queue()
        self.reader = ReaderProcess(self.read_trigger, self.read_return, db_dir, max_size)
        self.reader.start()
        ready = self.read_return.get(timeout=10.0)
        assert(ready in 'ready')
        self.write_queue = Queue()
        self.writer = WriterProcess(self.write_queue, db_dir, cache_quality)
        self.writer.start()
        self.hits = 0
        self.miss = 0
        self.size = image_size
        self.cache_quality = cache_quality


    def read(self, hs:int) -> np.ndarray:
        while not self.read_return.empty():#Ensure empty
            self.read_return.get_nowait()

        self.read_trigger.put_nowait(hs)
        try:
            ret = self.read_return.get(block=True, timeout=0.002) #Only wait for 1ms for read here
            if ret[0] != hs:
                print('Warning DB read slow! no match', self.miss, self.hits)
                self.miss += 1
                return None
            
            if ret[1] is None:
                self.miss += 1
                return None
            
            self.hits += 1
            if self.cache_quality != 100:
                buf = turbojpeg.decompress(ret[1], fastdct = True, fastupsample=True, pixelformat=turbojpeg.BGRA)
                return np.ndarray((self.size,self.size,4), dtype=np.uint8, buffer=buf)
            else:
                return np.frombuffer(ret[1], np.uint8).reshape((self.size,self.size,4))

        except queue.Empty:
            self.miss += 1
            print('Warning DB read slow!', self.miss, self.hits)
            return None

    def write(self, hs:int, data:np.ndarray):
        self.write_queue.put_nowait((hs, np.array(data)))

    def close(self):
        self.read_trigger.put_nowait(None)
        self.write_queue.put_nowait(None)
        time.sleep(1)
        self.write_queue.close()
        self.read_return.close()
        self.read_trigger.close()
        self.reader.terminate()
        self.writer.terminate()
        self.reader.join()
        self.writer.join()
        print('Finished close all')
    def __del__(self):
        self.close()

