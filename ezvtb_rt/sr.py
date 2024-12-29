import os
from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
import numpy as np
from ezvtb_rt.tha import THACore
from ezvtb_rt.rife import RIFECoreLinked

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

class SRCore():
    def __init__(self):
        pass
    def inference(self, return_now:bool) -> List[np.ndarray]:
        raise ValueError('No provided implementation')
    def fetchRes(self)->List[np.ndarray]:
        raise ValueError('No provided implementation')
    def viewRes(self)->List[np.ndarray]:
        raise ValueError('No provided implementation')

class SRLinkTha(SRCore):
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

    def inference(self, return_now:bool) -> List[np.ndarray]:
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
            return [self.memories['output_cv_img'].host]
        else:
            return None
        
    def fetchRes(self)->List[np.ndarray]:
        if self.returned == True:
            raise ValueError('Already fetched result')
        self.returned = True
        self.finishedFetch.synchronize()
        return [self.memories['output_cv_img'].host]
    
    def viewRes(self)->List[np.ndarray]:
        return [self.memories['output_cv_img'].host]
    

class SRLinkRife(SRCore):
    def __init__(self, model_dir:str, rife_core:RIFECoreLinked):
        self.instream = rife_core.instream
        self.scale = rife_core.scale
        self.fetchstream = cuda.Stream() 
        self.finishedExec = [cuda.Event() for _ in range(self.scale)]
        self.finishedFetch = cuda.Event() 
        self.returned = True
        self.engines = []
        self.memories = {}
        head_tail = os.path.split(model_dir)
        for i in range(self.scale):
            engine = Engine(head_tail[0], head_tail[1], 1)
            self.memories['framegen_'+str(i)] = rife_core.memories['framegen_'+str(i)]
            self.memories['output_'+str(i)] = createMemory(engine.outputs[0])
            engine.setInputMems([self.memories['framegen_'+str(i)]])
            engine.setOutputMems([self.memories['output_'+str(i)]]) 
            self.engines.append(engine)

    def inference(self, return_now:bool) -> List[np.ndarray]:
        self.fetchstream.synchronize()
        for i in range(len(self.engines)):
            #execution
            self.engines[i].exec(self.instream)
            self.finishedExec[i].record(self.instream)
            #Fetch to cpu
            self.fetchstream.wait_for_event(self.finishedExec[i])
            self.memories['output_'+str(i)].dtoh(self.fetchstream)
        self.finishedFetch.record(self.fetchstream)
        self.returned = False

        if return_now:
            self.returned = True
            self.finishedFetch.synchronize()
            return [self.memories['output_'+str(i)].host for i in range(self.scale)]
        else:
            return None
        
    def fetchRes(self)->List[np.ndarray]:
        if self.returned == True:
            raise ValueError('Already fetched result')
        self.returned = True
        self.finishedFetch.synchronize()
        return [self.memories['output_'+str(i)].host for i in range(self.scale)]
    
    def viewRes(self)->List[np.ndarray]:
        return [self.memories['output_'+str(i)].host for i in range(self.scale)]