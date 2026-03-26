from pathlib import Path
import hashlib
import os
import numpy as np
import tensorrt_rtx as trt
from typing import List, Dict, Tuple
import pycuda.driver as cuda
from os.path import join
import numpy
from tqdm import tqdm
from ezvtb_rt.init_utils import check_exist_all_models
import tempfile

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

def build_engine(onnx_file_path:str) -> bytes:
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(4)) # 4G
    config.tiling_optimization_level = trt.TilingOptimizationLevel.FULL
    config.num_compute_capabilities = 1
    config.set_compute_capability(trt.ComputeCapability.CURRENT, 0)
    
    def is_dynamic_shape()->bool:
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            dims = network.get_input(i).shape
            if dims[0] == -1:
                return True
        return False

    if is_dynamic_shape():
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            print('Setting dynamic shape for input:', input_name)
            dims = network.get_input(i).shape
            min_shape = trt.Dims(dims)
            opt_shape = trt.Dims(dims)
            max_shape = trt.Dims(dims)
            min_shape[0] = 1
            opt_shape[0] = 4
            max_shape[0] = 4
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            TRT_LOGGER.log(TRT_LOGGER.INFO, f'Setting dynamic shape for input {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}')
        config.add_optimization_profile(profile)
        config.builder_optimization_level = 5
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
    if path.endswith('.onnx'):
        # Create a cache directory in system temp
        cache_dir = os.path.join(tempfile.gettempdir(), 'ezvtuber_rt_engines')
        os.makedirs(cache_dir, exist_ok=True)

        # Generate cache filename from ONNX path (hash or based on filename)
        onnx_name = os.path.splitext(os.path.basename(path))[0]
        hash_suffix = hashlib.md5(path.encode('utf-8')).hexdigest()[:8]
        engine_path = os.path.join(cache_dir, f'{onnx_name}_{hash_suffix}.trt')

        if os.path.exists(engine_path):
            runtime = trt.Runtime(TRT_LOGGER)
            try:
                with open(engine_path, 'rb') as f:
                    engine_buffer = f.read()
                validity, diagnostics = runtime.get_engine_validity(engine_buffer)
            except Exception as exc:
                validity, diagnostics = trt.EngineValidity.INVALID, str(exc)

            if validity == trt.EngineValidity.INVALID:
                TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Cached engine {engine_path} is invalid. Rebuilding... Diagnostics: {diagnostics}')
                os.remove(engine_path)  # Remove invalid cache

        if not os.path.exists(engine_path):
            TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building engine from ONNX: {path}')
            engine = build_engine(path)
            save_engine(engine, engine_path)
        path = engine_path  # Use built trt engine path for loading
    TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')
    return engine
