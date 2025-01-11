from typing import List

import numpy as np
from ezvtb_rt.rife_ort import RIFEORTCore
from ezvtb_rt.tha_ort import THAORTCore, THAORTCoreNonDefault
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr_ort import SRORTCore
    
class CoreORT:
    def __init__(self, tha_path:str, rife_path:str = None, sr_path:str = None, device_id:int = 0, cacher:Cacher = None, use_eyebrow:bool = True):
        if device_id == 0:
            self.tha = THAORTCore(tha_path, use_eyebrow)
        else:
            self.tha = THAORTCoreNonDefault(tha_path, device_id, use_eyebrow)
        if rife_path is not None:
            self.rife = RIFEORTCore(rife_path, device_id)
            self.sr = None
        elif sr_path is not None:
            self.sr = SRORTCore(sr_path, device_id)
            self.rife = None
        else:
            self.rife = None
            self.sr = None
        self.cacher = cacher
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        pose = pose.astype(np.float32)

        if self.cacher is None:# Do not use cacher
            res = self.tha.inference(pose)
            if self.rife is not None:
                res = self.rife.inference(res)
            if self.sr is not None:
                res = self.sr.inference(res)
            return res
        
        #use cacher 
        hs = hash(str(pose))
        cached = self.cacher.read(hs)

        if cached is not None:# Cache hits
            if self.rife is not None: # There is rife
                return self.rife.inference([cached])
            elif self.sr is not None:
                return self.sr.inference([cached])
            else:
                return [cached]

        else: #cache missed
            res = self.tha.inference(pose)
            self.cacher.write(hs, res[0])
            if self.rife is not None:
                res = self.rife.inference(res)
            elif self.sr is not None:
                res = self.sr.inference(res)
            return res

