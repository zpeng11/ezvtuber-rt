import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import *
from os.path import join
from ezvtb_rt.engine import Engine, createMemory
from ezvtb_rt.rife import RIFECore
from ezvtb_rt.tha import THACore

class Core():
    def __init__(self, tha_model_dir, rift_model_dir):
        self.tha = THACore(tha_model_dir)
        self.rife = RIFECore(rift_model_dir, scale = -1, latest_frame=self.tha.memories['output_img'])
        
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()

        self.thaFinished = cuda.Event()
        self.rifeFinished = cuda.Event()
        self.resultFetchFinished = cuda.Event()

    def setImage(self, img:np.ndarray):
        assert(len(img.shape) == 4 and 
               img.shape[0] == 1 and 
               img.shape[1] == 4 and 
               img.shape[2] == 512 and 
               img.shape[3] == 512)
        np.copyto(self.tha.memories['input_img'].host, img)
        self.tha.memories['input_img'].htod(self.updatestream)
        self.tha.decomposer.exec(self.updatestream)
        self.updatestream.synchronize()

    def inference(self, pose:np.ndarray) -> List[np.ndarray]: # Put input and get last output

        self.outstream.wait_for_event(self.rifeFinished)
        for i in range(self.rife.scale -1):
            self.rife.memories['framegen_'+str(i)].dtoh(self.outstream)

        cuda.memcpy_dtod_async(self.rife.memories['old_frame'].device, self.tha.memories['output_img'].device, 
                               self.rife.memories['old_frame'].host.nbytes, self.outstream)
        self.resultFetchFinished.record(self.outstream)

        np.copyto(self.tha.memories['eyebrow_pose'].host, pose[:, :12])
        self.tha.memories['eyebrow_pose'].htod(self.instream)
        np.copyto(self.tha.memories['face_pose'].host, pose[:,12:12+27])
        self.tha.memories['face_pose'].htod(self.instream)
        np.copyto(self.tha.memories['rotation_pose'].host, pose[:,12+27:])
        self.tha.memories['rotation_pose'].htod(self.instream)
        # self.instream.wait_for_event(self.thaFinished)

        self.tha.combiner.exec(self.instream)
        self.tha.morpher.exec(self.instream)
        self.tha.rotator.exec(self.instream)
        
        self.tha.editor.exec(self.instream)

        self.instream.wait_for_event(self.resultFetchFinished)
        self.rife.engine.exec(self.instream)
        self.rifeFinished.record(self.instream)

        self.resultFetchFinished.synchronize()

        ret = []
        for i in range(self.rife.scale -1):
            ret.append(self.rife.memories['framegen_'+str(i)].host)

        ret.append(self.rife.memories['old_frame'])
        return ret
