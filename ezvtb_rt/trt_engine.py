"""TensorRT engine management module for real-time inference operations.

This module handles:
- TensorRT engine initialization
- Memory allocation for input/output tensors
- Execution context management
- Asynchronous inference operations
"""

from ezvtb_rt.trt_utils import *
from os.path import join

#memory management
class HostDeviceMem(object):
    def __init__(self, host_mem:numpy.ndarray, device_mem: cuda.DeviceAllocation):
        self.host: numpy.ndarray = host_mem
        self.device: cuda.DeviceAllocation = device_mem

    @classmethod
    def create(cls, shape, dtype):
        host_mem = cuda.pagelocked_empty(shape, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return cls(host_mem, device_mem)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    def __del__(self):
        self.device.free()

    def dtoh(self, stream:cuda.Stream):
        cuda.memcpy_dtoh_async(self.host, self.device, stream) 
    def htod(self, stream:cuda.Stream):
        cuda.memcpy_htod_async(self.device, self.host, stream)

    def bridgeFrom(self, other: 'HostDeviceMem', stream:cuda.Stream):
        assert self.host.nbytes == other.host.nbytes, f"Memory sizes must match for bridging, dst: {self.host.nbytes}, src: {other.host.nbytes}, dst shape: {self.host.shape}, src shape: {other.host.shape}"
        cuda.memcpy_dtod_async(self.device, other.device, self.host.nbytes, stream)



class TRTEngine:
    def __init__(self, engine_path: str, n_input:int):
        self.engine: trt.ICudaEngine = load_engine(engine_path)
        assert self.engine is not None, f'Failed to load engine from path {engine_path}'
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        self.engine_cache_path = engine_path + '.cache'
        self.cached_flag = False
        # create execution context
        self.runtime_config = self.engine.create_runtime_config()
        self.runtime_config.set_execution_context_allocation_strategy(trt.ExecutionContextAllocationStrategy.STATIC)
        self.runtime_cache = self.runtime_config.create_runtime_cache()
        if os.path.exists(self.engine_cache_path):
            with open(self.engine_cache_path, 'rb') as cache_file:
                cache_data = cache_file.read()
                self.runtime_cache.deserialize(cache_data)
                TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loaded runtime cache from {self.engine_cache_path}')
                self.cached_flag = True
        self.runtime_config.set_runtime_cache(self.runtime_cache)
        self.runtime_config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
        self.context: trt.IExecutionContext = self.engine.create_execution_context(self.runtime_config)
        self.n_batch: int = -1
        self.in_out_tensors: dict = {}
        self.inputs: List[HostDeviceMem] = []
        self.outputs: List[HostDeviceMem] = []
        
        # get input and output tensor names
        self.input_tensor_names: List[str] = [self.engine.get_tensor_name(i) for i in range(n_input)]
        self.output_tensor_names: List[str] = [self.engine.get_tensor_name(i) for i in range(n_input, self.engine.num_io_tensors)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input nodes: '+ str(self.input_tensor_names))
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output nodes: '+ str(self.output_tensor_names))

        self.input_tensors_original_shapes: Dict[str, List[int]] = {}
        self.output_tensors_original_shapes: Dict[str, List[int]] = {}
        for input_tensor_name in self.input_tensor_names:
            shape = [dim for dim in self.context.get_tensor_shape(input_tensor_name)]
            self.input_tensors_original_shapes[input_tensor_name] = shape
        for output_tensor_name in self.output_tensor_names:
            shape = [dim for dim in self.context.get_tensor_shape(output_tensor_name)]
            self.output_tensors_original_shapes[output_tensor_name] = shape

        # create stream
        self.stream: cuda.Stream = cuda.Stream()
        # For measuring inference time
        self.start_event: cuda.Event = cuda.Event()
        self.end_event: cuda.Event = cuda.Event()
            
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def putInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch: int = 1, stream: cuda.Stream = None, sync: bool = False):
        """Non-blocking input upload - doesn't synchronize stream"""
        stream = stream if stream is not None else self.stream
        input_tensors, output_tensors = self.configure_in_out_tensors(n_batch)
        for inp, inp_mem, inp_name in zip(np_inputs, input_tensors, self.input_tensor_names):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:', inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError(f'Input shape or type does not match for input tensor {inp_name}')
            if isinstance(inp, HostDeviceMem):
                cuda.memcpy_dtod_async(inp_mem.device, inp.device, 
                                   inp_mem.host.nbytes, stream)
            else:
                np.copyto(inp_mem.host, inp)
                inp_mem.htod(stream)
        if sync:
            stream.synchronize()

    def syncPutInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch:int = 1, stream: cuda.Stream = None):
        self.putInputs(np_inputs, n_batch, stream, sync=True)
    
    def asyncPutInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch:int = 1, stream: cuda.Stream = None):
        self.putInputs(np_inputs, n_batch, stream, sync=False)

    def kickoff(self, stream: cuda.Stream = None, sync:bool = False):
        if stream is None:
            stream = self.stream
        # Record the start event
        self.start_event.record(stream)
        # Run inference.
        self.context.execute_async_v3(stream.handle)
        # Record the end event
        self.end_event.record(stream)

        if not self.cached_flag:
            # serialize the cache into a memory blob and save to disk
            with open(self.engine_cache_path, 'wb') as runtime_cache_file:
                with self.runtime_cache.serialize() as buffer:
                    runtime_cache_file.write(buffer)
            TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saved runtime cache to {self.engine_cache_path}')
            self.cached_flag = True

        # Synchronize the stream if requested
        if sync:
            stream.synchronize()

    def asyncKickoff(self, stream: cuda.Stream = None):
        self.kickoff(stream, sync=False)

    def syncKickoff(self, stream: cuda.Stream = None):
        self.kickoff(stream, sync=True)

    # This is a numpy cpu interface for convenience
    def syncGetOutputs(self, copy:bool = True) -> List[np.ndarray]:
        for out_mem in self.outputs: out_mem.dtoh(self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        if copy:
            return [np.copy(outp.host) for outp in self.outputs]
        else:
            return [outp.host for outp in self.outputs]
    
    # This is a numpy cpu interface for convenience
    def syncInfer(self, np_inputs: List[np.ndarray], n_batch:int = 1) -> List[np.ndarray]:
        self.syncPutInputs(np_inputs, n_batch)
        self.asyncKickoff()
        return self.syncGetOutputs(True)
    
    def configure_in_out_tensors(self, n_batch:int = 1) -> Tuple[List[HostDeviceMem], List[HostDeviceMem]]:
        inputs: List[HostDeviceMem] = []
        outputs: List[HostDeviceMem] = []
        if  n_batch in self.in_out_tensors:
            inputs, outputs = self.in_out_tensors[n_batch]
            if n_batch != self.n_batch:
                # TRT_LOGGER.log(TRT_LOGGER.INFO, f'Reconfiguring tensor addresses for batch size {n_batch}')
                for input_tensor_name, input_mem in zip(self.input_tensor_names, inputs):
                    self.context.set_tensor_address(input_tensor_name, int(input_mem.device))
                    self.context.set_input_shape(input_tensor_name, trt.Dims(input_mem.host.shape))
                for output_tensor_name, output_mem in zip(self.output_tensor_names, outputs):
                    self.context.set_tensor_address(output_tensor_name, int(output_mem.device))
        else:
            for input_tensor_name in self.input_tensor_names:
                shape = self.input_tensors_original_shapes[input_tensor_name].copy()
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = n_batch
                dtype = trt.nptype(self.engine.get_tensor_dtype(input_tensor_name))
                # TRT_LOGGER.log(TRT_LOGGER.INFO, f'Allocating input tensor: {input_tensor_name} with shape {shape} and dtype {dtype}')
                mem = HostDeviceMem.create(shape, dtype)
                self.context.set_tensor_address(input_tensor_name, int(mem.device)) # Use this setup without binding for v3
                self.context.set_input_shape(input_tensor_name, trt.Dims(shape))
                inputs.append(mem)
            for output_tensor_name in self.output_tensor_names:
                shape = self.output_tensors_original_shapes[output_tensor_name].copy()
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = n_batch
                dtype = trt.nptype(self.engine.get_tensor_dtype(output_tensor_name))
                mem = HostDeviceMem.create(shape, dtype)
                self.context.set_tensor_address(output_tensor_name, int(mem.device)) # Use this setup without binding for v3
                outputs.append(mem)
            self.in_out_tensors[n_batch] = (inputs, outputs)
        self.n_batch = n_batch
        self.inputs = inputs
        self.outputs = outputs
        return inputs, outputs