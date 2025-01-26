from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory

class RIFESimple():
    """Simplified RIFE implementation for performance benchmarking.
    
    Provides basic frame interpolation using TensorRT engine with single CUDA stream.
    Optimized for measuring inference speed rather than production use."""
    def __init__(self, model_dir):
        super().__init__(model_dir)
        # Create CUDA stream for asynchronous operations
        self.instream = cuda.Stream()  # Primary stream for data transfers and execution

    def run(self, old_frame:np.ndarray, latest_frame:np.ndarray) -> List[np.ndarray]:
        """Process frames through TensorRT engine.
        
        Args:
            old_frame: Previous frame in NHWC format (batch, height, width, channels)
            latest_frame: Current frame in same format
            
        Returns:
            List of interpolated frames at different scales"""
        # Host-to-device memory transfers
        np.copyto(self.memories['old_frame'].host, old_frame)  # Copy CPU data to pinned host memory
        self.memories['old_frame'].htod(self.instream)  # Async H->D copy
        np.copyto(self.memories['latest_frame'].host, latest_frame)
        self.memories['latest_frame'].htod(self.instream)

        self.engine.exec(self.instream)  # Execute inference on current stream

        # Device-to-host transfers for each scale output
        for i in range(self.scale):
            self.memories['framegen_'+str(i)].dtoh(self.instream)  # Async D->H copy

        self.instream.synchronize()  # Wait for all async operations
        ret = []
        for i in range(self.scale):  # Collect results for each scale
            ret.append(self.memories['framegen_'+str(i)].host)  # Access host memory
        return ret
    
class RIFE():
    """Production RIFE implementation with pipelined execution.
    
    Uses dual CUDA streams to overlap compute and memory transfers.
    Supports multi-scale frame generation through model directory parsing."""
    def __init__(self, model_dir, instream=None, in_mem:HostDeviceMem=None):
        # Determine interpolation scale from model filename
        if 'x2' in model_dir:
            self.scale = 2  # 2x temporal interpolation
        elif 'x3' in model_dir:
            self.scale = 3  # 3x interpolation
        elif 'x4' in model_dir:
            self.scale = 4  # 4x interpolation
        else:
            raise ValueError('Model directory must contain x2/x3/x4 to indicate scale')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'Creating RIFE with scale {self.scale}')  # TensorRT initialization log

        self.engine = Engine(model_dir + '.trt', 2)  # Load TensorRT engine with 2MB workspace

        # Initialize memory buffers
        self.memories = {}
        # Input buffers
        self.memories['old_frame'] = createMemory(self.engine.inputs[0])  # Previous frame
        self.memories['latest_frame'] = in_mem if in_mem is not None else createMemory(self.engine.inputs[1])  # Reusable input memory
        # Output buffers for each scale
        for i in range(self.scale):
            self.memories['framegen_'+str(i)] = createMemory(self.engine.outputs[i])  # Allocate per-scale output
        
        # Bind memory to engine I/O
        self.engine.setInputMems([self.memories['old_frame'], self.memories['latest_frame']])  # Set input bindings
        outputs = [self.memories['framegen_'+str(i)] for i in range(self.scale)]
        self.engine.setOutputMems(outputs)

        # Create CUDA streams and events
        self.instream = instream if instream is not None else cuda.Stream()  # Inference stream
        self.copystream = cuda.Stream()  # Dedicated stream for memory copies
        self.finishedExec = cuda.Event()  # Synchronization event

    def inference(self):
        """Execute pipeline: 
        1. Wait for previous copies
        2. Run inference
        3. Signal completion
        4. Schedule frame buffer copy"""
        self.copystream.synchronize()  # Ensure previous copies complete
        self.engine.exec(self.instream)  # Run TensorRT engine
        self.finishedExec.record(self.instream)  # Mark inference completion

        # Schedule device-to-device copy after inference completes
        self.copystream.wait_for_event(self.finishedExec)  # Wait for inference finish
        cuda.memcpy_dtod_async(
            self.memories['old_frame'].device,  # dst: previous frame buffer
            self.memories['latest_frame'].device,  # src: current frame buffer
            self.memories['latest_frame'].host.nbytes,  # buffer size
            self.copystream  # Use copy stream to overlap with next inference
        )

    def fetchRes(self) -> List[np.ndarray]:
        """Retrieve and synchronize all results.
        
        Returns:
            List of processed frames in host memory"""
        for i in range(self.scale):
            self.memories['framegen_'+str(i)].dtoh(self.copystream)  # Schedule D->H copies
        ret = []
        self.copystream.synchronize()  # Wait for all copies
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)  # Access synchronized data
        return ret

    def viewRes(self) -> List[np.ndarray]:
        """Get current device memory state without synchronization.
        
        Returns:
            List of frames from device memory (may contain stale data)"""
        ret = []
        for i in range(self.scale):
            ret.append(self.memories['framegen_'+str(i)].host)  # Non-blocking access
        return ret
