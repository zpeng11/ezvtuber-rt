"""TensorRT engine management module for real-time inference operations.

This module handles:
- TensorRT engine initialization
- Memory allocation for input/output tensors
- Execution context management
- Asynchronous inference operations
"""

from ezvtb_rt.trt_utils import *
from os.path import join

class NodeInfo:
    """Stores metadata for inference engine tensor nodes.
    
    Attributes:
        name (str): Tensor name from TensorRT engine
        shape (List[int]): Tensor dimensions
        dtype (numpy.dtype): Data type of tensor elements
    """
    def __init__(self, name:str, shape:List[int], dtype):
        """Initialize tensor metadata container.
        
        Args:
            name: Tensor name from TensorRT engine
            shape: Dimensions of the tensor
            dtype: Data type of tensor elements
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def __str__(self) -> str:
        return self.name+": "+str(self.shape) + " "+str(self.dtype)
    def __repr__(self):
        return self.__str__()
    
def createMemory(nodeInfo : NodeInfo) -> HostDeviceMem:
    """Allocates page-locked host and device memory for a tensor.
    
    Args:
        nodeInfo: Tensor metadata specifying shape and data type
        
    Returns:
        HostDeviceMem: Paired host/device memory buffers
    """
    shape = nodeInfo.shape
    dtype = nodeInfo.dtype
    host_mem = cuda.pagelocked_empty(shape, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    return HostDeviceMem(host_mem, device_mem)

class Engine:
    """Manages TensorRT inference engine lifecycle and execution.
    
    Handles:
    - Engine/context initialization
    - Input/output tensor configuration
    - Memory binding
    - Asynchronous execution
    
    Args:
        model_path (str): Path to serialized TensorRT engine file
        n_inputs (int): Number of input tensors expected by the model
    """
    def __init__(self, model_path:str, n_inputs:int) -> None:
        """Initializes TensorRT engine and execution context."""
        self.engine = load_engine(model_path)
        assert(self.engine is not None), "Failed to load TensorRT engine - check model file path and compatibility"

        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        self.context = self.engine.create_execution_context()
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed inference context')
        
        self.inputs = []
        # Configure input tensors from engine metadata
        for i in range(n_inputs):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.inputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.inputs))

        self.outputs = []
        # Configure output tensors from engine metadata
        for i in range(n_inputs, self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.outputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.outputs))

    def setInputMems(self, inputMems:List[HostDeviceMem]) -> None:
        """Binds input memory buffers to engine tensors.
        
        Args:
            inputMems: List of pre-allocated host/device memory pairs
        """
        assert(len(inputMems) == len(self.inputs))
        for i in range(len(inputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.inputs[i].name)
            self.context.set_tensor_address(self.inputs[i].name, int(inputMems[i].device)) # Use this setup without binding for v3

    def setOutputMems(self, outputMems:List[HostDeviceMem]) -> None:
        """Binds output memory buffers to engine tensors.
        
        Args:
            outputMems: List of pre-allocated host/device memory pairs
        """
        assert(len(outputMems) == len(self.outputs))
        for i in range(len(outputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.outputs[i].name)
            self.context.set_tensor_address(self.outputs[i].name, int(outputMems[i].device)) # Use this setup without binding for v3
    def exec(self, stream) -> None:
        """Executes inference asynchronously using bound memory.
        
        Args:
            stream: CUDA stream for asynchronous execution
        """
        self.context.execute_async_v3(stream.handle)
