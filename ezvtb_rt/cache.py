from typing import Set
import numpy as np
from collections import OrderedDict
import os
from queue import Queue
import turbojpeg
from typing import List,Dict
import threading


def threadCompressSave(cache:OrderedDict, lock:threading.Lock, queue:Queue, max_size:int, cache_quality:int):
    max_kbytes = 1024
    cached_kbytes = 0
    while True:
        hs, data = queue.get(block=True)
        lock.acquire(blocking=True)
        cached = cache.get(hs)
        lock.release()
        if cached is not None:
            continue
        new_data = np.zeros((1024, 512, 4), data.dtype)
        new_data[:512, :, :] = data
        new_data[512:, :, 0] = data[:,:,3]

        if cache_quality == 100:
            compressed = turbojpeg.compress(new_data, cache_quality, turbojpeg.SAMP.Y420, lossless = True, fastdct = False, optimize= True, pixelformat=turbojpeg.BGRA)
        else:
            compressed = turbojpeg.compress(new_data, cache_quality, turbojpeg.SAMP.Y444, fastdct = False, optimize= True, pixelformat=turbojpeg.BGRA)
        lock.acquire(blocking=True)
        cache[hs] = compressed
        lock.release()
        print(len(compressed))
        cached_kbytes += len(compressed) /1024
        while cached_kbytes > max_kbytes:
            lock.acquire(blocking=True)
            poped = cache.popitem(last=False)
            lock.release()
            cached_kbytes -= len(poped[1])/1024
            poped = None


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
        self.last_hs = -1
    def read(self, hs:int) -> np.ndarray:
        self.lock.acquire(blocking=True)
        cached = self.cache.get(hs)
        self.lock.release()
        if self.continues_hits > 5:
            cached = None
        if cached is not None:
            if self.last_hs != hs:
                self.continues_hits += 1
            else:
                self.continues_hits = 0
            self.last_hs = hs
            self.hits += 1
            self.lock.acquire(blocking=True)
            self.cache.move_to_end(hs)
            self.lock.release()
            res = turbojpeg.decompress(cached, fastdct = False, fastupsample=False, pixelformat=turbojpeg.BGRA)
            decompressed_img =  np.ndarray((1024,512,4), dtype=np.uint8, buffer=res)
            result_img = decompressed_img[:512,:,:]
            result_img[:,:,3] = decompressed_img[512:,:,0]
            return result_img
        else:
            self.miss += 1
            self.continues_hits = 0
            self.last_hs = hs
            return None
    def write(self, hs:int, data:np.ndarray):
        self.queue.put_nowait((hs, data.copy()))
