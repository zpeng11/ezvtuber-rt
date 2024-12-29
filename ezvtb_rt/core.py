from ezvtb_rt.trt_utils import *
from ezvtb_rt.rife import RIFECore, RIFECoreLinked
from ezvtb_rt.tha import THACore
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr import SRCore

class Core():
    def __init__(self, cached_tha_core:THACore, cacher:Cacher = None, sr:SRCore = None, rife_core:RIFECoreLinked = None):
        self.tha = cached_tha_core
        self.rife = rife_core
        self.sr = sr
        self.cacher = cacher
    def setImage(self, img:np.ndarray):
        self.tha.setImage(img)

    def inference(self, pose:np.ndarray) -> List[np.ndarray]: #This numpy object should be copyed/used before next inference cycle
        last_component = self.tha
        if self.rife is not None:
            last_component = self.rife
        if self.sr is not None:
            last_component = self.sr

        if self.cacher is None: #Optional to disable cacher
            self.tha.inference(pose, False)
            if self.rife is not None:
                self.rife.inference(False)
            if self.sr is not None:
                self.sr.inference(False)

            return last_component.fetchRes()

        hs = hash(str(pose))
        cached = self.cacher.read(hs)

        if cached is not None: #Cache hits
            self.cacher.writeExecute()
            if self.cacher.cache_quality != 100:
                previous_res = last_component.viewRes()
                assert(len(previous_res) == len(cached))
                for i in range(len(previous_res)):
                    cached[i][:,:,3] = previous_res[i][:,:,3] #Use alpha channel from previous-calculated result because cacher does not store alpha if using turbojpeg
            return cached
        
        #Cache missed
        self.tha.inference(pose, False)
        if self.rife is not None:
            self.rife.inference(False)
        if self.sr is not None:
            self.sr.inference(False)
        self.cacher.writeExecute()
        result = last_component.fetchRes()
        self.cacher.write(hs, result)

        return result
