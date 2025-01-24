from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
from typing import List


class Affine:
    def __init__(self, model_dir:str, instream = None, linkedMemory = None):
        if '512' in model_dir:
            self.size = 512
        else:
            self.size = 1024
        
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating Affine engine')
        self.engine = Engine(model_dir, 4)
        self.memories = {}
        self.memories['input'] = linkedMemory if linkedMemory is not None else createMemory(self.engine.inputs[0])
        self.memories['angle'] = createMemory(self.engine.inputs[1])
        self.memories['translate'] = createMemory(self.engine.inputs[2])
        self.memories['scale'] = createMemory(self.engine.inputs[3])

        self.memories['output'] = createMemory(self.engine.outputs[0])

        self.engine.setInputMems([self.memories['input'], self.memories['angle'], self.memories['translate'], self.memories['output']])
        self.engine.setOutputMems([self.memories['output']])

        self.instream = instream if instream is not None else cuda.Stream() 
        self.outstream = cuda.Stream() 
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event()

    def inference(self):
        self.outstream.synchronize()
        self.engine.exec(self.instream)
        self.finishedExec.record(self.instream)

        self.outstream.wait_for_event(self.finishedExec)
        self.finishedFetch.record(self.outstream)

    def SyncfetchRes(self)->List[np.ndarray]:
        self.finishedFetch.synchronize()
        return [self.memories['output'].host]

    def viewRes(self)->List[np.ndarray]:
        return [self.memories['output'].host]