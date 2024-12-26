import os
from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
import numpy as np
from ezvtb_rt.tha import THACore

class SR():
    def __init__(self, model_dir):
        self.instream = cuda.Stream()
        head_tail = os.path.split(model_dir)
        self.engine = Engine(head_tail[0], head_tail[1], 1)
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
    
class SRLinked():
    def __init__(self, model_dir, tha_core:THACore):
        self.instream = tha_core.instream
        self.fetchstream = cuda.Stream() 
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event()
        self.returned = True
        head_tail = os.path.split(model_dir)
        self.engine = Engine(head_tail[0], head_tail[1], 1)
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
        self.returned = False

        if return_now:
            self.returned = True
            self.finishedFetch.synchronize()
            return self.memories['output_cv_img'].host
        else:
            return None
        
    def fetchRes(self)->np.ndarray:
        if self.returned == True:
            raise ValueError('Already fetched result')
        self.returned = True
        self.finishedFetch.synchronize()
        return self.memories['output_cv_img'].host