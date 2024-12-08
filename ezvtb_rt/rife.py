import os
from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
from ezvtb_rt.tha import THACore

class RIFECore:
    def __init__(self, model_dir:str, latest_frame:HostDeviceMem = None):
        if 'x2' in model_dir:
            self.scale = 2
        elif 'x3' in model_dir:
            self.scale = 3
        elif 'x4' in model_dir:
            self.scale = 4
        else:
            raise ValueError('can not determine scale')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'RIFE scale {self.scale}')
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating RIFE engine')
        self.prepareEngines(model_dir)
        self.prepareMemories(latest_frame)
        self.setMemsToEngines()

    def prepareEngines(self, model_dir:str, engineT = Engine): #inherit and pass different engine type
        head_tail = os.path.split(model_dir)
        self.engine = engineT(head_tail[0], head_tail[1], 2)
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
    def inference(self, return_now:bool) -> List[np.ndarray]:
        raise ValueError('No provided implementation')
    def fetchRes(self)->List[np.ndarray]:
        raise ValueError('No provided implementation')


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
    
class RIFECoreLinked(RIFECore):
    def __init__(self, model_dir, tha_core:THACore):
        super().__init__(model_dir, tha_core.memories['output_cv_img'])
        self.instream = tha_core.instream
        self.copystream = cuda.Stream() 
        self.finishedCopy = cuda.Event()
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event()
        self.returned = True

    def inference(self, return_now:bool) -> List[np.ndarray]:
        self.instream.wait_for_event(self.finishedCopy)
        self.engine.exec(self.instream)
        self.finishedExec.record(self.instream)

        for i in range(self.scale):
            self.memories['framegen_'+str(i)].dtoh(self.instream)
        self.finishedFetch.record(self.instream)

        self.copystream.wait_for_event(self.finishedExec)
        cuda.memcpy_dtod_async(self.memories['old_frame'].device, self.memories['latest_frame'].device, 
                                   self.memories['latest_frame'].host.nbytes, self.copystream)
        self.finishedCopy.record(self.copystream)
        self.returned = False

        if return_now:
            self.returned = True
            ret = []
            self.finishedFetch.synchronize()
            for i in range(self.scale):
                ret.append(self.memories['framegen_'+str(i)].host)
            return ret
        else:
            return None
    def fetchRes(self)->List[np.ndarray]:
        if self.returned == True:
            raise ValueError('Already fetched result')
        self.returned = True
        ret = []
        self.finishedFetch.synchronize()
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)
        return ret