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
        lock.acquire(blocking=True)
        cached = cache.get(hs)
        lock.release()
        if cached is not None:
            continue
        if cache_quality == 100:
            compressed = turbojpeg.compress(data, cache_quality, turbojpeg.SAMP.Y444, lossless = True, fastdct = False, optimize= True, pixelformat=turbojpeg.BGRA)
            alpha_channel = np.expand_dims(data[:,:,3], axis = 2)
            alpha_image = np.concatenate((alpha_channel,alpha_channel,alpha_channel), axis=2)
            compressed_alpha = turbojpeg.compress(alpha_image, 70, turbojpeg.SAMP.Y420, fastdct = False, optimize= True, pixelformat=turbojpeg.BGR)
            lock.acquire(blocking=True)
            cache[hs] = (compressed, compressed_alpha)
            lock.release()
            cached_kbytes += len(compressed) /1024
            cached_kbytes += len(compressed_alpha) /1024
            while cached_kbytes > max_kbytes:
                lock.acquire(blocking=True)
                poped = cache.popitem(last=False)
                lock.release()
                cached_kbytes -= poped[1][0].nbytes/1024
                cached_kbytes -= poped[1][1].nbytes/1024
                poped = None
        else:
            compressed = turbojpeg.compress(data, cache_quality, turbojpeg.SAMP.Y422, fastdct = False, optimize= True, pixelformat=turbojpeg.BGRA)
            alpha_channel = np.expand_dims(data[:,:,3], axis = 2)
            alpha_image = np.concatenate((alpha_channel,alpha_channel,alpha_channel), axis=2)
            compressed_alpha = turbojpeg.compress(alpha_image, 70, turbojpeg.SAMP.Y420, fastdct = False, optimize= True, pixelformat=turbojpeg.BGR)
            lock.acquire(blocking=True)
            cache[hs] = (compressed,compressed_alpha)
            lock.release()
            cached_kbytes += len(compressed) /1024
            cached_kbytes += len(compressed_alpha) /1024
            while cached_kbytes > max_kbytes:
                lock.acquire(blocking=True)
                poped = cache.popitem(last=False)
                lock.release()
                cached_kbytes -= len(poped[1][0])/1024
                cached_kbytes -= len(poped[1][1])/1024
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
            if self.cache_quality == 100:
                res = turbojpeg.decompress(cached[0], fastdct = False, fastupsample=False, pixelformat=turbojpeg.BGRA)
                res_alpha = turbojpeg.decompress(cached[1], fastdct = False, fastupsample=False, pixelformat=turbojpeg.BGR)
            else:
                res = turbojpeg.decompress(cached[0], fastdct = False, fastupsample=False, pixelformat=turbojpeg.BGRA)
                res_alpha = turbojpeg.decompress(cached[1], fastdct = False, fastupsample=False, pixelformat=turbojpeg.BGR)
            img =  np.ndarray((self.image_size,self.image_size,4), dtype=np.uint8, buffer=res)
            alpha_img = np.ndarray((self.image_size,self.image_size,3), dtype=np.uint8, buffer=res_alpha)
            img[:,:,3] = alpha_img[:,:,0]
            return img
        else:
            self.miss += 1
            self.continues_hits = 0
            self.last_hs = hs
            return None
    def write(self, hs:int, data:np.ndarray):
        self.queue.put_nowait((hs, data.copy()))
