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
    def __init__(self, model_dir:str, instream = None, in_mems:List[HostDeviceMem] = None):
        self.instream = instream
        self.scale = 1 if in_mems is None else len(in_mems)
        self.fetchstream = cuda.Stream() 
        self.finishedExec = [cuda.Event() for _ in range(self.scale)]
        self.engines = []
        self.memories = {}
        for i in range(self.scale):
            engine = Engine(model_dir, 1)
            self.memories['framegen_'+str(i)] = in_mems[i] if in_mems is not None else createMemory(engine.inputs[0])
            self.memories['output_'+str(i)] = createMemory(engine.outputs[0])
            engine.setInputMems([self.memories['framegen_'+str(i)]])
            engine.setOutputMems([self.memories['output_'+str(i)]]) 
            self.engines.append(engine)

    def inference(self):
        for i in range(len(self.engines)):
            #execution
            self.engines[i].exec(self.instream)
            self.finishedExec[i].record(self.instream)
            #Fetch to cpu
            self.fetchstream.wait_for_event(self.finishedExec[i])
            self.memories['output_'+str(i)].dtoh(self.fetchstream)
        
    def SyncfetchRes(self)->List[np.ndarray]:
        for i in range(len(self.engines)):
            self.fetchstream.wait_for_event(self.finishedExec[i])
            self.memories['output_'+str(i)].dtoh(self.fetchstream)
        self.fetchstream.synchronize()
        return [self.memories['output_'+str(i)].host for i in range(self.scale)]
    
    def viewRes(self)->List[np.ndarray]:
        return [self.memories['output_'+str(i)].host for i in range(self.scale)]