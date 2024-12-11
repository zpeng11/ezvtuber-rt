from typing import List

import numpy as np
from ezvtb_rt.rife_ort import RIFEORTCore
from ezvtb_rt.tha_ort import THAORTCore, THAORTCoreNonDefault
from ezvtb_rt.cache import Cacher

class CoreORT:
    def __init__(self, tha_path:str, rife_path:str, device_id:int = 0):
        if device_id == 0:
            self.tha = THAORTCore(tha_path)
        else:
            self.tha = THAORTCoreNonDefault(tha_path, device_id)
        self.rife = RIFEORTCore(rife_path, device_id)
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        tha_res = self.tha.inference(pose.astype(np.float32))
        return self.rife.inference(tha_res)
    
class CoreORTCached:
    def __init__(self, tha_path:str, rife_path:str = None, device_id:int = 0, cacher:Cacher = None):
        if device_id == 0:
            self.tha = THAORTCore(tha_path)
        else:
            self.tha = THAORTCoreNonDefault(tha_path, device_id)
        if rife_path is not None:
            self.rife = RIFEORTCore(rife_path, device_id)
        else:
            self.rife = None
        self.cacher = cacher
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        if self.cacher is None:
            if self.rife is None:
                return [self.tha.inference(pose.astype(np.float32))]
            else:
                tha_res = self.tha.inference(pose.astype(np.float32))
                return self.rife.inference(tha_res)
        else:
            hs = hash(frozenset(pose.flatten()))
            cached = self.cacher.read(hs)
            if cached is not None:
                if self.cacher.cache_quality != 100:
                    cached[:,:,3] = self.last_tha_res[:,:,3]
                if self.rife is None:
                    return [cached]
                else:
                    return self.rife.inference(cached)
            else:
                if self.rife is None:
                    tha_res = self.tha.inference(pose.astype(np.float32))
                    self.cacher.write(hs, tha_res)
                    return [tha_res]
                else:
                    tha_res = self.tha.inference(pose.astype(np.float32))
                    self.cacher.write(hs, tha_res)
                    return self.rife.inference(tha_res)

