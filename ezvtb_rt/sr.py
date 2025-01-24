from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory, HostDeviceMem
import numpy as np


class SRSimple():
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

class SR():
    def __init__(self, model_dir, instream = None, in_mem:HostDeviceMem = None):
        self.instream = instream if instream is not None else cuda.Stream() 
        self.fetchstream = cuda.Stream() 
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event()
        self.engine = Engine(model_dir, 1)
        self.memories = {}
        self.memories['input'] = in_mem if in_mem is not None else createMemory(self.engine.inputs[0])
        self.memories['output'] = createMemory(self.engine.outputs[0])
        self.engine.setInputMems([self.memories['input']])
        self.engine.setOutputMems([self.memories['output']]) 

    def inference(self):
        self.fetchstream.synchronize()
        self.engine.exec(self.instream)
        self.finishedExec.record(self.instream)

        self.fetchstream.wait_for_event(self.finishedExec)
        self.memories['output'].dtoh(self.fetchstream)
        self.finishedFetch.record(self.fetchstream)
        
    def SyncfetchRes(self)->List[np.ndarray]:
        self.finishedFetch.synchronize()
        return [self.memories['output'].host]
    
    def viewRes(self)->List[np.ndarray]:
        return [self.memories['output'].host]