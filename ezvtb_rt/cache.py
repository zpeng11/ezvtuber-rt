from typing import Set
import numpy as np
from collections import OrderedDict
import os
from queue import Queue
import turbojpeg
from typing import List,Dict
import threading


def threadCompressSave(cache:OrderedDict, lock:threading.Lock, queue:Queue, max_size:int, cache_quality:int):
    max_kbytes = max_size * 1024 * 1024
    cached_kbytes = 0
    while True:
        hs, data = queue.get(block=True)
        collect = []
        if cache_quality == 100:
            for item in data:
                collect.append(np.array(item))
                cached_kbytes += item.nbytes /1024
            while cached_kbytes > max_kbytes:
                lock.acquire(blocking=True)
                poped = cache.popitem(last=False)
                lock.release()
                for item in poped[1]:
                    cached_kbytes -= item.nbytes/1024
                poped = None
        else:
            for item in data:
                compressed = turbojpeg.compress(item, cache_quality, turbojpeg.SAMP.Y420,fastdct = True, optimize= True, pixelformat=turbojpeg.BGRA)
                collect.append(compressed)
                cached_kbytes += len(compressed) /1024
            while cached_kbytes > max_kbytes:
                lock.acquire(blocking=True)
                poped = cache.popitem(last=False)
                lock.release()
                for item in poped[1]:
                    cached_kbytes -= len(item)/1024
                poped = None
        lock.acquire(blocking=True)
        cache[hs] = collect
        lock.release()


class Cacher:
    def __init__(self, max_size:float = 2.0, cache_quality:int = 90,  image_size:int = 512): #Size in GBs
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.queue = Queue()

        self.hits = 0
        self.miss = 0
        self.image_size = image_size
        self.cache_quality = cache_quality

        self.thread = threading.Thread(target=threadCompressSave, args=(self.cache, self.lock, self.queue, max_size, cache_quality), daemon=True)
        self.thread.start()
        self.temp_data = None
        self.continues_hits = 0
    def read(self, hs:int) -> List[np.ndarray]:
        self.lock.acquire(blocking=True)
        cached = self.cache.get(hs)
        self.lock.release()
        if self.continues_hits > 5:
            cached = None
        if cached is not None and len(cached)>0:
            self.hits += 1
            self.continues_hits += 1
            self.lock.acquire(blocking=True)
            self.cache.move_to_end(hs)
            self.lock.release()
            if self.cache_quality == 100:
                imgs = cached
            else:
                imgs = []
                for item in cached:
                    res = turbojpeg.decompress(item, fastdct = True, fastupsample=True, pixelformat=turbojpeg.BGRA)
                    imgs.append(np.ndarray((self.image_size,self.image_size,4), dtype=np.uint8, buffer=res))
            return imgs
        else:
            self.miss += 1
            self.continues_hits = 0
            return None
    def write(self, hs:int, data:List[np.ndarray]):
        self.temp_data = (hs, data)
    def writeExecute(self): #Copy is time consuming
        if self.temp_data is None:
            return
        data_copied = []
        for item in self.temp_data[1]:
            data_copied.append(item.copy())
        self.queue.put_nowait((self.temp_data[0], data_copied))
        self.temp_data = None
