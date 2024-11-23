import sys
import os
sys.path.append(os.getcwd())
from typing import Set
import cv2
import sqlite3
from abc import ABC
import numpy as np
from collections import OrderedDict
import zlib
import os
from multiprocessing import Process, Queue, shared_memory
import multiprocessing as mp
import queue

class Cacher(ABC):
    def read(hs:int) -> np.ndarray:
        pass
    def write(hs:int, data:np.ndarray):
        pass

def THARes2JPEG(res:np.ndarray, jpeg_quality:int = 90) -> bytes:
    saveInt = ((res / 2.0 + 0.5) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    shapes = res.shape
    reshaped = saveInt.reshape(shapes[1],shapes[2] * shapes[3]).transpose().reshape(shapes[2], shapes[3], shapes[1])
    cvimg = cv2.cvtColor(reshaped, cv2.COLOR_RGB2BGR)
    if jpeg_quality < 100:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ret = zlib.compress(cv2.imencode('.jpg', cvimg, encode_param)[1].tobytes())
    else:
        ret = zlib.compress(cv2.imencode('.png', cvimg)[1].tobytes())
    return ret


def JPEG2THARes(data:bytes , dtype:np.dtype = np.float16)->np.ndarray: 
    cvimg = cv2.imdecode(np.fromstring(zlib.decompress(data), dtype="uint8") , cv2.IMREAD_COLOR)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    shapes = cvimg.shape
    reshaped = cvimg.reshape(shapes[0] * shapes[1], shapes[2]).transpose().reshape(1, shapes[2], shapes[0], shapes[1])
    totype = reshaped.astype(dtype)
    return (totype / 255.0 - 0.5) * 2.0
    

class RAMCacher(Cacher):
    def __init__(self, max_size:int, dtype:np.dtype = np.float16, quality:int = 90): #Size in GBs
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
        self.dtype = dtype
        self.quality = quality
    def read(self, hs:int) -> np.ndarray:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            img = JPEG2THARes(cached, self.dtype)
            return img
        else:
            self.miss += 1
            return None
    def write(self, hs:int, data:np.ndarray) -> tuple[int, np.ndarray]:
        self.cache[hs] = THARes2JPEG(data, self.quality)
        if len(self.cache) * len(self.cache[hs]) > self.max_size * 1024 * 1024 * 1024:
            return self.cache.popitem(last=False)



class ReaderProcess(Process): # A seperate process that reads input from database, run in another process to avoid potential block in sqlite operation
    def __init__(self, read_trigger:Queue, 
                 read_return:Queue, 
                 db_path:str, max_size:int, #DB max ram size
                 dtype:np.dtype = np.float16
                 ):
        super().__init__()
        self.read_trigger = read_trigger
        self.read_return = read_return
        self.dtype = dtype
        self.db_path = db_path
        self.max_size = max_size

    def run(self):
        self.conn = self.create_db_for_read(self.db_path, self.max_size)
        self.read_return.put_nowait('ready')
        while True:
            try:
                hs = self.read_trigger.get(block=True, timeout=100.0)
            except queue.Empty:
                hs = None

            if hs is None or hs is not int:
                break
            ret = self.conn.execute('SELECT * FROM cache WHERE hash=?', (hs,)).fetchone()
            if not ret:
                self.read_return.put_nowait((hs, None))
            else:
                img = JPEG2THARes(ret[1], self.dtype)
                self.read_return.put_nowait((hs, img))
        self.conn.close()

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
        return conn
                

class WriterProcess(Process): # A seperate process that write input from database, run in another process to avoid potential block in sqlite operation
    def __init__(self, write_queue:Queue, db_path:str, max_size:int):
        self.write_queue = write_queue
        self.db_path = db_path
        self.max_size = max_size

    def run(self):
        assert(os.path.isfile(self.db_path)) # Must already exist
        self.conn = self.db_for_write(self.db_path, self.max_size)
        while True:
            try:
                res = self.write_queue.get(block=True, timeout=100.0)
            except queue.Empty:
                res = None
            if res is None or res is not tuple:
                break
            hs = res[0]
            data = THARes2JPEG(res[1])
            self.conn.execute('INSERT OR IGNORE INTO cache(hash, bytes) VALUES (?, ?)',(hs, data))
            self.conn.commit()
        self.conn.close()

    def db_for_write(self, db_path:str, max_size:int):
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode = wal;')
        conn.execute('PRAGMA synchronous = normal;')
        conn.execute('PRAGMA temp_store = memory;')
        conn.execute(f'PRAGMA mmap_size = {str(1024 * 1024 * 1024 * max_size/2)};')
        conn.execute(f'PRAGMA cache_size = -{str(1024 * 1024 * max_size/2)};')
        return conn



class DBCacherMP(Cacher):
    def __init__(self, max_size:int = 1,  db_dir:str = '.', dtype:np.dtype = np.float16) -> None:
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, 'cacher.sqlite')
        self.read_trigger = Queue()
        self.read_return = Queue()
        self.reader = ReaderProcess(self.read_trigger, self.read_return, db_path, max_size, dtype)
        self.reader.start()
        ready = self.read_return.get(timeout=1.0)
        assert(ready in 'ready')
        self.write_queue = Queue()
        self.writer = WriterProcess(self.write_queue, db_path, max_size, max_size)
        self.writer.start()
        self.hits = 0
        self.miss = 0


    def read(self, hs:int) -> np.ndarray:
        self.read_trigger.put_nowait(hs)
        try:
            ret = self.read_return.get(block=True, timeout=0.01) #Only wait for 10ms for read here
            while(ret[0] is int and ret[0] != hs):
                ret = self.read_return.get(block=True, timeout=0.01) # This works for removing missed read
            if ret[1] is None:
                self.miss += 1
            else:
                self.hits += 1
            return ret[1]
        except queue.Empty:
            return None

    def write(self, hs:int, data:np.ndarray):
        self.write_queue.put_nowait((hs, data))

    def close(self):
        self.read_trigger.put_nowait(None)
        self.write_queue.put_nowait(None)
    def __del__(self):
        self.close()

