from typing import List, Optional
import numpy as np
from ezvtb_rt.rife_ort import RIFEORT
from ezvtb_rt.tha_ort import THAORT, THAORTNonDefault
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr_ort import SRORT
from ezvtb_rt.common import Core
    
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
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        pose = pose.astype(np.float32)

        if self.cacher is None:# Do not use cacher
            res = self.tha.inference(pose)
        else:
            #use cacher 
            hs = hash(str(pose))
            cached = self.cacher.read(hs)

            if cached is not None:# Cache hits
                res = [cached]
            else: #cache missed
                res = self.tha.inference(pose)
                self.cacher.write(hs, res[0])

        if self.rife is not None:
            res = self.rife.inference(res)
        if self.sr is not None:
            res = self.sr.inference(res)
        return res

