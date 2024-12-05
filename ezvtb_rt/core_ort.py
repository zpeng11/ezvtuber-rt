import sys
import os
sys.path.append(os.getcwd())
from typing import List

import numpy as np
from ezvtb_rt.rife_ort import RIFEORTCore
from ezvtb_rt.tha_ort import THAORTCore, THAORTCoreNonDefault

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

    