import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import *
from ezvtb_rt.rife import RIFECore, RIFECoreLinked
from ezvtb_rt.tha import THACore
from ezvtb_rt.cache import Cacher

class Core():
    def __init__(self, tha_model_dir:str, rift_model_dir:str):
        self.tha = THACore(tha_model_dir)
        self.rife = RIFECore(rift_model_dir, latest_frame=self.tha.memories['output_cv_img'])
        
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        
        self.rifeFinished = cuda.Event()
        self.resultFetchFinished = cuda.Event()

    def setImage(self, img:np.ndarray):
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4
               )
        np.copyto(self.tha.memories['input_img'].host, img)
        self.tha.memories['input_img'].htod(self.updatestream)
        self.tha.decomposer.exec(self.updatestream)
        self.updatestream.synchronize()

    def inference(self, pose:np.ndarray) -> List[np.ndarray]: # Put input and get output of previous framegen

        self.outstream.wait_for_event(self.rifeFinished)
        for i in range(self.rife.scale):
            self.rife.memories['framegen_'+str(i)].dtoh(self.outstream)

        self.resultFetchFinished.record(self.outstream)

        cuda.memcpy_dtod_async(self.rife.memories['old_frame'].device, self.tha.memories['output_cv_img'].device, 
                               self.rife.memories['old_frame'].host.nbytes, self.instream)

        np.copyto(self.tha.memories['eyebrow_pose'].host, pose[:, :12])
        self.tha.memories['eyebrow_pose'].htod(self.instream)
        np.copyto(self.tha.memories['face_pose'].host, pose[:,12:12+27])
        self.tha.memories['face_pose'].htod(self.instream)
        np.copyto(self.tha.memories['rotation_pose'].host, pose[:,12+27:])
        self.tha.memories['rotation_pose'].htod(self.instream)

        self.tha.combiner.exec(self.instream)
        self.tha.morpher.exec(self.instream)
        self.tha.rotator.exec(self.instream)
        self.tha.editor.exec(self.instream)

        self.instream.wait_for_event(self.resultFetchFinished)
        self.rife.engine.exec(self.instream)
        self.rifeFinished.record(self.instream)

        self.resultFetchFinished.synchronize()

        ret = []
        for i in range(self.rife.scale):
            ret.append(self.rife.memories['framegen_'+str(i)].host)
        return ret

class CoreCached():
    def __init__(self, cached_tha_core:THACore, cacher:Cacher, rife_core:RIFECoreLinked):
        self.tha = cached_tha_core
        self.cacher = cacher
        self.rife = rife_core
    def setImage(self, img:np.ndarray):
        self.tha.setImage(img)
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        if self.cacher is None: #Optional to disable cacher
            self.tha.inference(pose, False)
            return self.rife.inference(True)

        hs = hash(str(pose))
        cached = self.cacher.read(hs)
        if cached is None:
            self.tha.inference(pose, False)
            self.rife.inference(False)
        else:
            pass
        self.rife.inference(False)
