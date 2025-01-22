import os
from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
import numpy as np
from ezvtb_rt.tha import THACore
from ezvtb_rt.rife import RIFECoreLinked

class SR():
    def __init__(self, model_dir):
        self.instream = cuda.Stream()
        self.engine = Engine(model_dir, 1)
        self.memories = {}
        self.memories['input'] = createMemory(self.engine.inputs[0])
        self.memories['output'] = createMemory(self.engine.outputs[0])
        self.engine.setInputMems([self.memories['input']])
        self.engine.setOutputMems([self.memories['output']])
        
        self.returned = True

    def run(self, img:np.ndarray) -> np.ndarray:
        np.copyto(self.memories['input'].host, img)
        self.memories['input'].htod(self.instream)
        self.engine.exec(self.instream)
        self.memories['output'].dtoh(self.instream)
        self.instream.synchronize()
        return self.memories['output'].host

class SRCore():
    def __init__(self):
        pass
    def inference(self):
        raise ValueError('No provided implementation')
    def SyncfetchRes(self)->np.ndarray:
        raise ValueError('No provided implementation')
    def viewRes(self)->np.ndarray:
        raise ValueError('No provided implementation')

class SRLinkTha(SRCore):
    def __init__(self, model_dir, tha_core:THACore):
        self.instream = tha_core.instream
        self.fetchstream = cuda.Stream() 
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event()
        self.returned = True
        self.engine = Engine(model_dir, 1)
        self.memories = {}
        self.memories['input'] = tha_core.memories['output_cv_img']
        self.memories['output_cv_img'] = createMemory(self.engine.outputs[0])
        self.engine.setInputMems([self.memories['input']])
        self.engine.setOutputMems([self.memories['output_cv_img']]) 

    def inference(self, return_now:bool) -> np.ndarray:
        self.fetchstream.synchronize()
        self.engine.exec(self.instream)
        self.finishedExec.record(self.instream)

        self.fetchstream.wait_for_event(self.finishedExec)
        self.memories['output_cv_img'].dtoh(self.fetchstream)
        self.finishedFetch.record(self.fetchstream)
        
    def SyncfetchRes(self)->np.ndarray:
        self.finishedFetch.synchronize()
        return self.memories['output_cv_img'].host
    
    def viewRes(self)->np.ndarray:
        return self.memories['output_cv_img'].host
    

class SRLinkRife(SRCore):
    def __init__(self, model_dir:str, rife_core:RIFECoreLinked, idx:int):
        self.instream = rife_core.instream
        self.scale = rife_core.scale
        assert(idx < self.scale)
        self.fetchstream = cuda.Stream() 
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event() 
        self.returned = True
        self.engine = Engine(model_dir, 1)
        self.memories = {}
        self.memories['input'] = rife_core.memories['framegen_'+str(idx)]
        self.memories['output_cv_img'] = createMemory(self.outputs[0])
        self.engine.setInputMems([self.memories['input']])
        self.engine.setOutputMems([self.memories['output_cv_img']]) 

    def inference(self):
        self.fetchstream.synchronize()
        self.engine.exec(self.instream)
        self.finishedExec.record(self.instream)
            #Fetch to cpu
        self.fetchstream.wait_for_event(self.finishedExec)
        self.memories['output_cv_img'].dtoh(self.fetchstream)
        self.finishedFetch.record(self.fetchstream)
        
    def SyncfetchRes(self)->np.ndarray:
        self.finishedFetch.synchronize()
        return self.memories['output_cv_img'].host
    
    def viewRes(self)->np.ndarray:
        return self.memories['output_cv_img'].host