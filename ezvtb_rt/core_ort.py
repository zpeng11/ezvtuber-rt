from typing import List, Optional

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from ezvtb_rt.rife_ort import RIFEORT
from ezvtb_rt.tha_ort import THAORT, THAORTNonDefault
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr_ort import SRORT
from ezvtb_rt.common import Core
from queue import Queue
import threading
    
class CoreORT(Core):
    def __init__(self, tha_path:Optional[str] = None, rife_path:Optional[str] = None, sr_path:Optional[str] = None, device_id:int = 0, cache_max_volume:float = 2.0, cache_quality:int = 90, use_eyebrow:bool = True):
        if device_id == 0:
            self.tha = THAORT(tha_path, use_eyebrow)
        else:
            self.tha = THAORTNonDefault(tha_path, device_id, use_eyebrow)

        self.rife = None
        self.sr = None
        self.cacher = None

        if rife_path is not None:
            self.rife = RIFEORT(rife_path, device_id)
        if sr_path is not None:
            self.sr = SRORT(sr_path, device_id)
        if cache_max_volume > 0.0:
            self.cacher = Cacher(cache_max_volume, cache_quality)

        result_shape = 512 if self.sr is None else 1024
        self.scale = 1 if self.rife is None else self.rife.scale
        self.result = [np.zeros((result_shape, result_shape, 4),dtype=np.uint8) for _ in range(self.scale)]
        self.result_ptr = 0

        self.in_queue = Queue()
        self.out_queue = Queue()
        self.thread = threading.Thread(target=self.run_thread, args=(), daemon=True)
        self.thread.start()

    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)

    def run_thread(self):
        self.out_queue.put_nowait(self.result)
        while True:
            self.endCount()
            pose = self.in_queue.get(block=True, timeout=None)
            self.startCount()
            if self.cacher is None:# Do not use cacher
                res = self.tha.inference(pose)
                if self.rife is not None:
                    res = self.rife.inference(res)
                if self.sr is not None:
                    res = self.sr.inference(res)
                self.out_queue.put_nowait(res)
                continue

            #use cacher 
            hs = hash(str(pose))
            cached = self.cacher.read(hs)

            if cached is not None:# Cache hits
                res = [cached]
                if self.rife is not None: # There is rife
                    res = self.rife.inference(res)
                if self.sr is not None:
                    res = self.sr.inference(res)
                self.out_queue.put_nowait(res)

            else: #cache missed
                res = self.tha.inference(pose)
                self.cacher.write(hs, res[0])
                if self.rife is not None:
                    res = self.rife.inference(res)
                if self.sr is not None:
                    res = self.sr.inference(res)
                self.out_queue.put_nowait(res)

    def infer(self, pose:np.ndarray):
        self.result =  self.out_queue.get(block=True, timeout=None)
        self.result_ptr = 0
        pose = pose.astype(np.float32)
        self.in_queue.put_nowait(pose)

    def finishedFetch(self) -> bool:
        return self.result_ptr >= self.scale
    
    def syncFetchRes(self) -> np.ndarray:
        ret = self.result[self.result_ptr]
        self.result_ptr += 1
        return ret
    
    def averageInterval(self) -> float:
        if len(self.intervals) >= 1:
            return self.intervalSum / len(self.intervals) / self.scale
        else:
            return 0.001

if __name__ == '__main__':
    import cv2
    import sys
    

    from tqdm import tqdm
    input = np.zeros((1,45), dtype= np.float32)
    core = CoreORT('C:\\EasyVtuber\\data\\models\\tha3\\seperable\\fp32', 'C:\\EasyVtuber\\data\\models\\rife_512\\x4\\fp32', cache_max_volume = 0)
    for i in tqdm(range(1000)):
        core.syncFetchRes()
        core.syncFetchRes()
        core.syncFetchRes()
        core.syncFetchRes()
        core.infer(input)
    core = CoreORT('C:\\EasyVtuber\\data\\models\\tha3\\seperable\\fp32', cache_max_volume = 0)
    for i in tqdm(range(1000)):
        core.syncFetchRes()
        core.infer(input)