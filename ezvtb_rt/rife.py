import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import *
from os.path import join
from ezvtb_rt.engine import Engine, createMemory

class RIFECore:
    def __init__(self, model_dir:str, scale:int = -1, latest_frame:HostDeviceMem = None):
        if scale < 2:
            if 'x2' in model_dir:
                self.scale = 2
            elif 'x3' in model_dir:
                self.scale = 3
            elif 'x4' in model_dir:
                self.scale = 4
            else:
                raise ValueError('can not determine scale')
        else:
            self.scale = scale
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'RIFE scale {self.scale}')
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating RIFE engine')
        self.prepareEngines(model_dir)
        self.prepareMemories(latest_frame)
        self.setMemsToEngines()

    def prepareEngines(self, model_dir:str, engineT = Engine): #inherit and pass different engine type
        self.engine = engineT(model_dir, 'rife', 2)
    def prepareMemories(self, latest_frame:HostDeviceMem): 
        self.memories = {}
        self.memories['old_frame'] = createMemory(self.engine.inputs[0])
        if latest_frame is None:
            self.memories['latest_frame'] = createMemory(self.engine.inputs[1])
        else:
            self.memories['latest_frame'] = latest_frame
        for i in range(self.scale):
            self.memories['framegen_'+str(i)] = createMemory(self.engine.outputs[i])
    def setMemsToEngines(self):
        self.engine.setInputMems([self.memories['old_frame'], self.memories['latest_frame']])
        outputs = [self.memories['framegen_'+str(i)] for i in range(self.scale)]
        self.engine.setOutputMems(outputs)


class RIFECoreSimple(RIFECore): #Simple implementation of tensorrt rife core, just for benchmarking rife's performance on given platform
    def __init__(self, model_dir):
        super().__init__(model_dir)
        # create stream
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        # Create CUDA events
        self.finishedFetchRes = cuda.Event()
        self.finishedExec = cuda.Event()


    def run(self, old_frame:np.ndarray, latest_frame:np.ndarray) -> List[np.ndarray]: # Give new input and return last result at the same time

        self.outstream.wait_for_event(self.finishedExec)
        
        for i in range(self.scale):
            self.memories['framegen_'+str(i)].dtoh(self.outstream)
        
        self.finishedFetchRes.record(self.outstream)

        np.copyto(self.memories['old_frame'].host, old_frame)
        self.memories['old_frame'].htod(self.instream)
        np.copyto(self.memories['latest_frame'].host, latest_frame)
        self.memories['latest_frame'].htod(self.instream)

        self.instream.wait_for_event(self.finishedFetchRes)

        self.engine.exec(self.instream)
    
        self.finishedExec.record(self.instream)

        self.finishedFetchRes.synchronize()

        ret = []
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)
        return ret
    
    
