import sys
import os
sys.path.append(os.getcwd())
from typing import Set
import cv2
import sqlite3

class DBCachePool:
    def __init__(self, dir:str, memory_cache:int = 1) -> None:
        os.makedirs(dir, exist_ok=True)
        self.memory_cache = min(memory_cache, 8) #8Gb max
        self.dir = dir
        self.db_connects = []
        self.writing_db = None
        for filename in os.listdir(dir):
            if filename.endswith(".sqlite") and filename.split('.')[0].isnumeric():
                db_path = os.path.join(dir, filename)
                self.db_connects.append(self.create_or_access_db(db_path))
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

