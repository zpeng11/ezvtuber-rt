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

class Cacher(ABC):
    def read(hs:int) -> np.ndarray:
        pass
    def write(hs:int, data:np.ndarray):
        pass

def THARes2JPEG(res:np.ndarray, jpeg_quality:int = 100) -> bytes:
    saveInt = ((res / 2.0 + 0.5) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    reshaped = saveInt.reshape(3,512 * 512).transpose().reshape(512, 512, 3)
    cvimg = cv2.cvtColor(reshaped, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    return zlib.compress(cv2.imencode('.jpg', cvimg, encode_param)[1].tobytes())


def JPEG2THARes(data:bytes)->np.ndarray: 
    cvimg = cv2.imdecode(np.fromstring(zlib.decompress(data), dtype="uint8") , cv2.IMREAD_COLOR)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    reshaped = cvimg.reshape(512 * 512, 3).transpose().reshape(1, 3, 512, 512)
    tofp16 = reshaped.astype(np.float16)
    return (tofp16 / 255.0 - 0.5) * 2.0
    

class RAMCacher(Cacher):
    def __init__(self, max_size:int): #Size in GBs
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
    def read(self, hs:int) -> tuple[bool, np.ndarray]:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            img = JPEG2THARes(cached)
            return img
        else:
            self.miss += 1
            return None
    def write(self, hs:int, data:np.ndarray) -> tuple[int, np.ndarray]:
        self.cache[hs] = THARes2JPEG(data)
        if len(self.cache) * len(self.cache[hs]) > self.max_size * 1024 * 1024 * 1024:
            return self.cache.popitem(last=False)



class DBCacher(Cacher):
    def __init__(self, model_dir:str = '.', max_size:int = 1) -> None:
        os.makedirs(dir, exist_ok=True)
        self.max_size = max_size
        self.db_path = os.path.join(self.dir, 'Cacher.sqlite')
        self.db_connect = self.create_or_access_db(db_path)
        #TO change below
        if len(self.db_connects) == 0:
            self.create_new_db()
        self.writing_db = self.db_connects[-1]

    def create_or_access_db(self, db_path:str):
        conn = sqlite3.connect(db_path)
        # conn.execute('PRAGMA journal_mode = wal2;')
        # conn.execute(f'PRAGMA journal_size_limit = {str(1024 * 1024 * 4)}') #4Mb
        # conn.execute('PRAGMA synchronous = normal;')
        conn.execute('PRAGMA temp_store = memory;')
        conn.execute('PRAGMA page_size = 4096;') # 4Kb Windows optimization
        conn.execute(f'PRAGMA mmap_size = -{str(1024 * 1024 * self.memory_cache)};')
        conn.execute(f'PRAGMA cache_size = -{str(1024 * 1024 * self.memory_cache)};')
        # Create the table
        conn.execute('CREATE TABLE IF NOT EXISTS cache( hash INTEGER PRIMARY KEY, bytes BLOB NOT NULL);')
        return conn

    def create_new_db(self):
        db_path = os.path.join(self.dir, str(len(self.db_connects))+ '.sqlite')
        assert(not os.path.exists(db_path)) # New db should not exist
        self.db_connects.append(self.create_or_access_db(db_path))
        self.writing_db = self.db_connects[-1]
        for conn in self.db_connects:
            conn.execute(f'PRAGMA mmap_size = -{str(1024 * 1024 * self.memory_cache//len(self.db_connects))};')
            conn.execute(f'PRAGMA cache_size = -{str(1024 * 1024 * self.memory_cache//len(self.db_connects))};')

    def read(self, hs:int) -> Set:
        for conn in self.db_connects:
            ret = conn.execute('SELECT * FROM cache WHERE hash=?', (hs,)).fetchone()
            if ret:
                return (True, ret[1])
        return (False,)
    def write(self, hs:int, data:bytes):
        count = self.writing_db.execute('PRAGMA application_id ').fetchone()[0]
        if count * len(data) > 1024 * 1024 * 1024 * 8: #Over 4GB sqlite gets pretty slow when querying
            self.create_new_db()
            count = 0
        self.writing_db.execute(f'PRAGMA application_id = {count + 1}')
        self.writing_db.execute('INSERT OR IGNORE INTO cache(hash, bytes) VALUES (?, ?)',(hs, data))

    def close(self):
        for conn in self.db_connects:
            conn.commit()
            conn.close()
    def __del__(self):
        self.close()

