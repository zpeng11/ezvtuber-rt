from ezvtb_rt.trt_utils import *
from os.path import join

class NodeInfo:
    def __init__(self, name:str, shape:List[int], dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def __str__(self) -> str:
        return self.name+": "+str(self.shape) + " "+str(self.dtype)
    def __repr__(self):
        return self.__str__()
    
def createMemory(nodeInfo : NodeInfo):
    shape = nodeInfo.shape
    dtype = nodeInfo.dtype
    host_mem = cuda.pagelocked_empty(shape, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    return HostDeviceMem(host_mem, device_mem)

class Engine:
    def __init__(self, model_dir:str, model_component:str, n_inputs:int):
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {join(model_dir, model_component)}')
        self.engine = get_trt_engine(model_dir, model_component)
        assert(self.engine is not None)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')

        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        self.context = self.engine.create_execution_context()
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed inference context')
        
        self.inputs = []
        for i in range(n_inputs):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.inputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.inputs))

        self.outputs = []
        for i in range(n_inputs, self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.outputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.outputs))

    def setInputMems(self, inputMems:List[HostDeviceMem]): # inputs pass by reference
        assert(len(inputMems) == len(self.inputs))
        for i in range(len(inputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.inputs[i].name)
            self.context.set_tensor_address(self.inputs[i].name, int(inputMems[i].device)) # Use this setup without binding for v3

    def setOutputMems(self, outputMems:List[HostDeviceMem]): # outputs pass by reference
        assert(len(outputMems) == len(self.outputs))
        for i in range(len(outputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.outputs[i].name)
            self.context.set_tensor_address(self.outputs[i].name, int(outputMems[i].device)) # Use this setup without binding for v3
    def exec(self, stream):
        self.context.execute_async_v3(stream.handle)