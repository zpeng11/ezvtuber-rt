from pathlib import Path
import os
import numpy as np
import tensorrt as trt
from typing import List
import pycuda.driver as cuda
from os.path import join
import numpy
from tqdm import tqdm
from ezvtb_rt.init_utils import check_exist_all_models

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Solution from https://github.com/NVIDIA/TensorRT/issues/1050#issuecomment-775019583
def cudaSetDevice(device_idx):
    from ctypes import cdll, c_char_p
    libcudart = cdll.LoadLibrary('cudart64_12.dll')
    libcudart.cudaGetErrorString.restype = c_char_p
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + str(error_string))

def check_build_all_models(data_dir:str = '') -> bool:
    all_models_list = check_exist_all_models(data_dir)
    #Check TRT support
    print('Start testing if TensorRT works on this machine')
    try:
        for fullpath in tqdm(all_models_list):
            dir, filename = os.path.split(fullpath)
            trt_filename = filename.split('.')[0] + '.trt'
            trt_fullpath = os.path.join(dir, trt_filename)
            if os.path.isfile(trt_fullpath):
                try:
                    if load_engine(trt_fullpath) is not None:
                        continue
                    else:
                        print(f'Can not successfully load {trt_fullpath}, build again')
                except:
                    print(f'Can not successfully load {trt_fullpath}, build again')
            dtype = 'fp16' if 'fp16' in fullpath else 'fp32'
            engine_seri = build_engine(fullpath, dtype)
            if engine_seri is None:
                print('Error while building engine')
                return False
            save_engine(engine_seri, trt_fullpath)
    except:
        return False
    return True


def build_engine(onnx_file_path:str, precision:str):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        parse_res = parser.parse(model.read())
        if not parse_res:
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')
    def GiB(val):
        return val * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1)) # 1G
    
    if precision == 'fp32':
        config.set_flag(trt.BuilderFlag.TF32)
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        raise ValueError('precision must be one of fp32 or fp16')
        # Build engine.
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    return serialized_engine

def save_engine(engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')
def load_engine(path):
    TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')
    return engine

#memory management
class HostDeviceMem(object):
    def __init__(self, host_mem:numpy.ndarray, device_mem: cuda.DeviceAllocation):
        self.host = host_mem
        self.device = device_mem

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

class Processor:
    def __init__(self, engine: trt.ICudaEngine, n_input:int):
        self.engine = engine
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        # create execution context
        self.context = engine.create_execution_context()
        
        # get input and output tensor names
        self.input_tensor_names = [engine.get_tensor_name(i) for i in range(n_input)]
        self.output_tensor_names = [engine.get_tensor_name(i) for i in range(n_input, self.engine.num_io_tensors)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input nodes: '+ str(self.input_tensor_names))
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output nodes: '+ str(self.output_tensor_names))

        #create memories and bindings
        self.inputs = []
        self.outputs = []
        for bindingName in engine:
            shape = [dim for dim in self.context.get_tensor_shape(bindingName)]
            dtype = trt.nptype(engine.get_tensor_dtype(bindingName))
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            mem = HostDeviceMem(host_mem, device_mem)
            self.context.set_tensor_address(bindingName, int(device_mem)) # Use this setup without binding for v3
            if bindingName in self.input_tensor_names:
                self.inputs.append(mem)
            else:
                self.outputs.append(mem)

        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

            
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def loadInputs(self, inputs: List[np.ndarray]):
        # set input shapes, the output shapes are inferred automatically
        for inp, inp_mem in zip(inputs, self.inputs):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:',inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError('Input shape or type does not match')
            np.copyto(inp_mem.host, inp)
        for inp_mem in self.inputs: inp_mem.htod(self.stream)
        # Synchronize the stream
        self.stream.synchronize()

    def kickoff(self):
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)
        # Synchronize the stream
        self.stream.synchronize()

    def extractOutputs(self, copy:bool = True) -> List[np.ndarray]:
        for out_mem in self.outputs: out_mem.dtoh(self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        if copy:
            return [np.copy(outp.host) for outp in self.outputs]
        else:
            return [outp.host for outp in self.outputs]
        
        
    def inference(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        # set input shapes, the output shapes are inferred automatically
        for inp, inp_mem in zip(inputs, self.inputs):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:',inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError('Input shape or type does not match')
            np.copyto(inp_mem.host, inp)

        for inp_mem in self.inputs: inp_mem.htod(self.stream)
            
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)

        for out_mem in self.outputs: out_mem.dtoh(self.stream)
            
        # Synchronize the stream
        self.stream.synchronize()
        
        return [np.copy(outp.host) for outp in self.outputs]
    
def get_trt_engine(model_dir:str, component:str):
    trt_filename = join(model_dir, component+'.trt')
    if os.path.isfile(trt_filename):
        return load_engine(trt_filename)
    else:
        raise ValueError(f'TRT has not built for {trt_filename}, call check_build_all_models() first')

