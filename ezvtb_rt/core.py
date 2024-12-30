from ezvtb_rt.trt_utils import *
from ezvtb_rt.rife import RIFECore, RIFECoreLinked
from ezvtb_rt.tha import THACore
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr import SRCore

class Core():
    def __init__(self, cached_tha_core:THACore, cacher:Cacher = None, sr:SRCore = None, rife_core:RIFECoreLinked = None):
        self.tha = cached_tha_core
        if sr is not None:
            self.rife = None
            self.sr = sr
        elif rife_core is not None:
            self.rife = rife_core
            self.sr = None
        else:
            self.rife = None
            self.sr = None
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
            
            if self.rife is not None:
                if self.cacher.cache_quality != 100:
                    self.tha.memories['output_cv_img'].host[:,:,:3] = cached[:,:,:3]
                else:
                    np.copyto(self.tha.memories['output_cv_img'].host, cached)
                self.tha.memories['output_cv_img'].htod(self.tha.instream)
                return self.rife.inference(True)
            elif self.sr is not None:
                if self.cacher.cache_quality != 100:
                    self.tha.memories['output_cv_img'].host[:,:,:3] = cached[:,:,:3]
                else:
                    np.copyto(self.tha.memories['output_cv_img'].host, cached)
                self.tha.memories['output_cv_img'].htod(self.tha.instream)
                return self.sr.inference(True)
            else:
                if self.cacher.cache_quality != 100:
                    tha_res = self.tha.viewRes() 
                    cached[:,:,3] = tha_res[0][:,:,3] #Use alpha channel from previous-calculated result because cacher does not store alpha if using turbojpeg
                return [cached]

        
        #Cache missed
        self.tha.inference(pose, False)
        if self.rife is not None:
            self.rife.inference(False)
            tha_res = self.tha.fetchRes()[0]
            self.cacher.write(hs, tha_res)
            return self.rife.fetchRes()
        elif self.sr is not None:
            self.sr.inference(False)
            tha_res = self.tha.fetchRes()[0]
            self.cacher.write(hs, tha_res)
            return self.sr.fetchRes()
        else:
            tha_res = self.tha.fetchRes()[0]
            self.cacher.write(hs, tha_res)
            return [tha_res]
