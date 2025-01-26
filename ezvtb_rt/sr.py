from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory, HostDeviceMem
import numpy as np


class SRSimple():
    """Single-stream super-resolution processor using TensorRT engine."""
    
    def __init__(self, model_dir):
        """Initialize CUDA stream, TensorRT engine and memory buffers.
        
        Args:
            model_dir: Path to TensorRT engine file (without .trt extension)
        """
        # Create dedicated CUDA stream and load TensorRT engine
        self.instream = cuda.Stream()
        self.engine = Engine(model_dir + '.trt', 1)
        
        # Allocate input/output memory buffers
        self.memories = {}
        self.memories['input'] = createMemory(self.engine.inputs[0])
        self.memories['output'] = createMemory(self.engine.outputs[0])
        
        # Configure engine memory bindings
        self.engine.setInputMems([self.memories['input']])
        self.engine.setOutputMems([self.memories['output']])
        
        self.returned = True  # Track processing state

    def run(self, img:np.ndarray) -> np.ndarray:
        """Execute super-resolution processing pipeline.
        
        Args:
            img: Input low-resolution image as numpy array
            
        Returns:
            High-resolution output image as numpy array
        """
        # Copy input data to host buffer
        np.copyto(self.memories['input'].host, img)
        
        # Transfer input to device memory
        self.memories['input'].htod(self.instream)
        
        # Execute TensorRT inference
        self.engine.exec(self.instream)
        
        # Transfer output back to host memory
        self.memories['output'].dtoh(self.instream)
        
        # Wait for all operations to complete
        self.instream.synchronize()
        
        return self.memories['output'].host
    
class SR():
    """Multi-stream parallel super-resolution processor."""
    
    def __init__(self, model_dir:str, instream = None, in_mems:List[HostDeviceMem] = None):
        """Initialize parallel processing infrastructure.
        
        Args:
            model_dir: Path to TensorRT engine file (without .trt extension)
            instream: Optional shared CUDA stream for execution
            in_mems: Optional pre-allocated input memory buffers
        """
        self.instream = instream  # Shared CUDA stream
        self.scale = 1 if in_mems is None else len(in_mems)  # Number of parallel streams
        self.fetchstream = cuda.Stream()  # Dedicated stream for data transfers
        self.finishedExec = [cuda.Event() for _ in range(self.scale)]  # Sync events
        self.engines = []  # TensorRT engine instances
        self.memories = {}  # Memory buffers
        
        # Initialize each processing stream
        for i in range(self.scale):
            # Load TensorRT engine
            engine = Engine(model_dir + '.trt', 1)
            
            # Configure memory buffers (reuse existing or create new)
            self.memories[f'framegen_{i}'] = in_mems[i] if in_mems else createMemory(engine.inputs[0])
            self.memories[f'output_{i}'] = createMemory(engine.outputs[0])
            
            # Bind engine memory
            engine.setInputMems([self.memories[f'framegen_{i}']])
            engine.setOutputMems([self.memories[f'output_{i}']])
            
            self.engines.append(engine)

    def inference(self):
        """Execute parallel inference across all engines."""
        for i in range(len(self.engines)):
            # Execute engine and record completion event
            self.engines[i].exec(self.instream)
            self.finishedExec[i].record(self.instream)
        
    def fetchRes(self)->List[np.ndarray]:
        """Retrieve processed results from all streams.
        
        Returns:
            List of super-resolved output images
        """
        for i in range(len(self.engines)):
            # Wait for engine completion before transferring
            self.fetchstream.wait_for_event(self.finishedExec[i])
            # Copy output from device to host memory
            self.memories[f'output_{i}'].dtoh(self.fetchstream)
            
        # Ensure all transfers complete
        self.fetchstream.synchronize()
        
        return [self.memories[f'output_{i}'].host for i in range(self.scale)]
    
    def viewRes(self)->List[np.ndarray]:
        """Get current output buffers without synchronization.
        
        Note: May return stale data if inference not complete
        
        Returns:
            List of output buffers in current state
        """
        return [self.memories[f'output_{i}'].host for i in range(self.scale)]
