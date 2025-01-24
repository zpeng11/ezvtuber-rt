from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory

class RIFESimple(): #Simple implementation of tensorrt rife core, just for benchmarking rife's performance on given platform
    def __init__(self, model_dir):
        super().__init__(model_dir)
        # create stream
        self.instream = cuda.Stream()

    def run(self, old_frame:np.ndarray, latest_frame:np.ndarray) -> List[np.ndarray]: 
        np.copyto(self.memories['old_frame'].host, old_frame)
        self.memories['old_frame'].htod(self.instream)
        np.copyto(self.memories['latest_frame'].host, latest_frame)
        self.memories['latest_frame'].htod(self.instream)

        self.engine.exec(self.instream)

        for i in range(self.scale):
            self.memories['framegen_'+str(i)].dtoh(self.instream)

        self.instream.synchronize()
        ret = []
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)
        return ret
    
class RIFE():
    def __init__(self, model_dir, instream = None, in_mem:HostDeviceMem = None):
        if 'x2' in model_dir:
            self.scale = 2
        elif 'x3' in model_dir:
            self.scale = 3
        elif 'x4' in model_dir:
            self.scale = 4
        else:
            raise ValueError('can not determine scale')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'Creating RIFE with scale {self.scale}')

        self.engine = Engine(model_dir, 2)

        #Prepare input outputs memories
        self.memories = {}
        self.memories['old_frame'] = createMemory(self.engine.inputs[0])
        self.memories['latest_frame'] = in_mem if in_mem is not None else createMemory(self.engine.inputs[1])
        for i in range(self.scale):
            self.memories['framegen_'+str(i)] = createMemory(self.engine.outputs[i])
        
        #bind memories
        self.engine.setInputMems([self.memories['old_frame'], self.memories['latest_frame']])
        outputs = [self.memories['framegen_'+str(i)] for i in range(self.scale)]
        self.engine.setOutputMems(outputs)

        self.instream = instream if instream is not None else cuda.Stream() 
        self.copystream = cuda.Stream() 
        self.finishedExec = cuda.Event()
        self.finishedFetch = cuda.Event()

    def inference(self):
        self.copystream.synchronize()
        self.engine.exec(self.instream)
        self.finishedExec.record(self.instream)

        self.copystream.wait_for_event(self.finishedExec)
        for i in range(self.scale):
            self.memories['framegen_'+str(i)].dtoh(self.copystream)
        self.finishedFetch.record(self.copystream)
        cuda.memcpy_dtod_async(self.memories['old_frame'].device, self.memories['latest_frame'].device, 
                                   self.memories['latest_frame'].host.nbytes, self.copystream)

    def SyncfetchRes(self)->List[np.ndarray]:
        ret = []
        self.finishedFetch.synchronize()
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)
        return ret

    def viewRes(self)->List[np.ndarray]:
        ret = []
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)
        return ret