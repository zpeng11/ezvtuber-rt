from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
from collections import OrderedDict

# THASimple - A basic implementation of THA core using TensorRT
# Used primarily for benchmarking performance on different platforms
class THASimple():
    """A simplified TensorRT-based implementation of the THA core for performance benchmarking
    
    Attributes:
        updatestream (cuda.Stream): CUDA stream for image update operations
        instream (cuda.Stream): CUDA stream for inference operations
        outstream (cuda.Stream): CUDA stream for result fetching
        finishedFetchRes (cuda.Event): CUDA event signaling result fetch completion
        finishedExec (cuda.Event): CUDA event signaling inference completion
    """
    def __init__(self, model_dir):
        """Initialize THASimple with model directory and CUDA resources"""
        super().__init__(model_dir)
        # create stream
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        # Create a CUDA events
        self.finishedFetchRes = cuda.Event()
        self.finishedExec = cuda.Event()
    
    def setImage(self, img:np.ndarray):
        """Set input image and prepare for processing
        Args:
            img (np.ndarray): Input image array (512x512x4 RGBA format)
        """
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4)
        np.copyto(self.memories['input_img'].host, img)
        self.memories['input_img'].htod(self.updatestream)
        self.decomposer.exec(self.updatestream)
        self.updatestream.synchronize()
    def inference(self, pose:np.ndarray) -> np.ndarray:
        """Run inference pipeline with pose data
        Args:
            pose (np.ndarray): Input pose parameters array
        Returns:
            np.ndarray: Processed output image from previous inference
        """
        
        self.outstream.wait_for_event(self.finishedExec)
        self.memories['output_cv_img'].dtoh(self.outstream)
        self.finishedFetchRes.record(self.outstream)

        np.copyto(self.memories['eyebrow_pose'].host, pose[:, :12])
        self.memories['eyebrow_pose'].htod(self.instream)
        np.copyto(self.memories['face_pose'].host, pose[:,12:12+27])
        self.memories['face_pose'].htod(self.instream)
        np.copyto(self.memories['rotation_pose'].host, pose[:,12+27:])
        self.memories['rotation_pose'].htod(self.instream)

        self.combiner.exec(self.instream)
        self.morpher.exec(self.instream)
        self.rotator.exec(self.instream)
        self.instream.wait_for_event(self.finishedFetchRes)
        self.editor.exec(self.instream)
        self.finishedExec.record(self.instream)
        
        self.finishedFetchRes.synchronize()
        return self.memories['output_cv_img'].host

# VRAMMem - Manages allocation and deallocation of GPU memory
# Wraps CUDA memory allocation with automatic cleanup
class VRAMMem(object):
    """Manages allocation and lifecycle of CUDA device memory"""
    def __init__(self, nbytes:int):
        self.device = cuda.mem_alloc(nbytes)

    def __str__(self):
        return "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    def __del__(self):
        self.device.free()


class VRAMCacher(object):
    """Implements LRU cache strategy for GPU memory management
    Attributes:
        pool (list): Available memory sets
        cache (OrderedDict): LRU cache of memory sets
        hits/miss (int): Cache performance metrics
    """
    def __init__(self, nbytes1:int, nbytes2:int, max_size:float):
        sum_nkbytes = (nbytes1 + nbytes2)/1024
        self.pool = []
        while len(self.pool) * sum_nkbytes < max_size * 1024 * 1024:
            self.pool.append((VRAMMem(nbytes1), VRAMMem(nbytes2)))
        self.nbytes1 = nbytes1
        self.nbytes2 = nbytes2
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
        if max_size <= 0:
            self.single_mem = (VRAMMem(nbytes1), VRAMMem(nbytes2)) #for zero memory cache
        self.max_size = max_size
    def query(self, hs:int)->bool:
        cached = self.cache.get(hs)
        if cached is not None:
            return True
        else:
            return False
    def read_mem_set(self, hs:int)->set[VRAMMem, VRAMMem]:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            return cached
        else:
            self.miss += 1
            return None
    def write_mem_set(self, hs:int)->set[VRAMMem, VRAMMem]:
        if self.max_size <= 0: #Do this to handle zero memory cache
            return self.single_mem
        if len(self.pool) != 0:
            mem_set = self.pool.pop()
        else:
            mem_set = self.cache.popitem(last=False)[1]
        self.cache[hs] = mem_set
        return mem_set

class THA():
    """Optimized THA implementation with GPU memory caching
    Attributes:
        combiner_cacher (VRAMCacher): Cache for eyebrow combination results
        morpher_cacher (VRAMCacher): Cache for face morphing results
        streams (cuda.Stream): Dedicated CUDA streams for different operations
        events (cuda.Event): Synchronization events for pipeline stages
    """
    def __init__(self, model_dir, vram_cache_size:float = 1.0, use_eyebrow:bool = True):
        """Initialize THA with model directory and caching configuration
        Args:
            model_dir (str): Directory containing TensorRT engine files
            vram_cache_size (float): Total GPU memory allocated for caching (in MB)
            use_eyebrow (bool): Enable eyebrow pose processing
        """
        self.prepareEngines(model_dir)
        self.prepareMemories()
        self.setMemsToEngines()
        if use_eyebrow:
            self.combiner_cacher = VRAMCacher(self.memories['eyebrow_image'].host.nbytes, 
                                            self.memories['morpher_decoded'].host.nbytes, 
                                            0.1 * vram_cache_size)
        else:
            self.combiner_cacher = None
        
        self.morpher_cacher = VRAMCacher(self.memories['face_morphed_full'].host.nbytes,
                                         self.memories['face_morphed_half'].host.nbytes,
                                         (0.9 if use_eyebrow else 1.0) * vram_cache_size)
        self.use_eyebrow = use_eyebrow

        # create stream
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.cachestream = cuda.Stream()
        self.outstream = cuda.Stream()
        # Create a CUDA events
        self.finishedMorpher = cuda.Event()
        self.finishedCombiner = cuda.Event()
        self.finishedFetch = cuda.Event()
        self.finishedExec = cuda.Event()
    
    def setImage(self, img:np.ndarray):
        """Prepare input image and run initial processing
        Args:
            img (np.ndarray): Input image array (512x512x4 RGBA format)
        """
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4)
        np.copyto(self.memories['input_img'].host, img)
        self.memories['input_img'].htod(self.updatestream)
        self.decomposer.exec(self.updatestream)
        if not self.use_eyebrow:
            self.memories['eyebrow_pose'].host[:,:] = 0.0
            self.memories['eyebrow_pose'].htod(self.updatestream)
            self.combiner.exec(self.updatestream)
        self.updatestream.synchronize()

    def inference(self, pose:np.ndarray):
        """Execute full inference pipeline with pose data and caching
        Args:
            pose (np.ndarray): Combined pose parameters array containing:
                - eyebrow_pose: First 12 elements
                - face_pose: Next 27 elements 
                - rotation_pose: Remaining elements
        """
        eyebrow_pose = pose[:, :12]
        face_pose = pose[:,12:12+27]
        rotation_pose = pose[:,12+27:]

        np.copyto(self.memories['rotation_pose'].host, rotation_pose)
        self.memories['rotation_pose'].htod(self.instream)

        morpher_hash = hash(str(pose[0,:12+27]))
        morpher_cached = self.morpher_cacher.read_mem_set(morpher_hash)
        combiner_hash = hash(str(pose[0,:12]))
        if self.use_eyebrow:
            combiner_cached = self.combiner_cacher.read_mem_set(combiner_hash)
        else:
            combiner_cached = None

        self.cachestream.synchronize()
        self.outstream.synchronize()
        if(morpher_cached is not None):
            cuda.memcpy_dtod_async(self.memories['face_morphed_full'].device, morpher_cached[0].device, 
                                   self.memories['face_morphed_full'].host.nbytes, self.instream)
            cuda.memcpy_dtod_async(self.memories['face_morphed_half'].device, morpher_cached[1].device, 
                                   self.memories['face_morphed_half'].host.nbytes, self.instream)
            self.rotator.exec(self.instream)
            self.editor.exec(self.instream)
            self.finishedExec.record(self.instream)
        elif(combiner_cached is not None or not self.use_eyebrow):
            #prepare input
            if self.use_eyebrow:
                cuda.memcpy_dtod_async(self.memories['eyebrow_image'].device, combiner_cached[0].device, 
                                    self.memories['eyebrow_image'].host.nbytes, self.instream)
                cuda.memcpy_dtod_async(self.memories['morpher_decoded'].device, combiner_cached[1].device, 
                                    self.memories['morpher_decoded'].host.nbytes, self.instream)
            np.copyto(self.memories['face_pose'].host, face_pose)
            self.memories['face_pose'].htod(self.instream)
            
            #Execute morpher
            self.morpher.exec(self.instream)
            self.finishedMorpher.record(self.instream)
            
            #Execute the rest
            self.rotator.exec(self.instream)
            self.editor.exec(self.instream)
            self.finishedExec.record(self.instream)

            #cache morpher result
            self.cachestream.wait_for_event(self.finishedMorpher)
            morpher_cache_write = self.morpher_cacher.write_mem_set(morpher_hash)
            cuda.memcpy_dtod_async(morpher_cache_write[0].device, self.memories['face_morphed_full'].device, 
                                   self.memories['face_morphed_full'].host.nbytes, self.cachestream)
            cuda.memcpy_dtod_async(morpher_cache_write[1].device, self.memories['face_morphed_half'].device, 
                                   self.memories['face_morphed_half'].host.nbytes, self.cachestream)
        else:
            #prepare input
            np.copyto(self.memories['face_pose'].host, face_pose)
            self.memories['face_pose'].htod(self.instream)
            np.copyto(self.memories['eyebrow_pose'].host, eyebrow_pose)
            self.memories['eyebrow_pose'].htod(self.instream)

            #execute combiner
            self.combiner.exec(self.instream)
            self.finishedCombiner.record(self.instream)
            
            #execute morpher
            self.morpher.exec(self.instream)
            self.finishedMorpher.record(self.instream)

            #execute the rest
            self.rotator.exec(self.instream)
            self.editor.exec(self.instream)
            self.finishedExec.record(self.instream)

            #cache morpher result
            combiner_cache_write = self.combiner_cacher.write_mem_set(combiner_hash)
            self.cachestream.wait_for_event(self.finishedCombiner)
            cuda.memcpy_dtod_async(combiner_cache_write[0].device, self.memories['eyebrow_image'].device, 
                                   self.memories['eyebrow_image'].host.nbytes, self.cachestream)
            cuda.memcpy_dtod_async(combiner_cache_write[1].device, self.memories['morpher_decoded'].device, 
                                   self.memories['morpher_decoded'].host.nbytes, self.cachestream)
            
            #cache morpher result
            morpher_cache_write = self.morpher_cacher.write_mem_set(morpher_hash)
            self.cachestream.wait_for_event(self.finishedMorpher)
            cuda.memcpy_dtod_async(morpher_cache_write[0].device, self.memories['face_morphed_full'].device, 
                                   self.memories['face_morphed_full'].host.nbytes, self.cachestream)
            cuda.memcpy_dtod_async(morpher_cache_write[1].device, self.memories['face_morphed_half'].device, 
                                   self.memories['face_morphed_half'].host.nbytes, self.cachestream)
        
        self.outstream.wait_for_event(self.finishedExec)
        self.memories['output_cv_img'].dtoh(self.outstream)
        self.finishedFetch.record(self.outstream)

    def fetchRes(self)->List[np.ndarray]:
        """Retrieve processed results from GPU
        Returns:
            List[np.ndarray]: List containing output image array
        """
        self.finishedFetch.synchronize()
        return [self.memories['output_cv_img'].host]
    def viewRes(self)->List[np.ndarray]:
        """Get current output without synchronization
        Returns:
            List[np.ndarray]: List containing most recent output image
        """
        return [self.memories['output_cv_img'].host]


    def prepareEngines(self, model_dir):
        """Load and initialize TensorRT engines from model directory"""
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating Engines')
        self.decomposer = Engine(join(model_dir, 'decomposer.trt'), 1)
        self.combiner = Engine(join(model_dir, 'combiner.trt'), 4)
        self.morpher = Engine(join(model_dir, 'morpher.trt'), 4)
        self.rotator = Engine(join(model_dir, 'rotator.trt'), 2)
        self.editor = Engine(join(model_dir, 'editor.trt'), 4)

    def prepareMemories(self):
        """Allocate GPU memory for all engine inputs/outputs"""
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating memories on VRAM')
        self.memories = {}
        self.memories['input_img'] = createMemory(self.decomposer.inputs[0])
        self.memories["background_layer"] = createMemory(self.decomposer.outputs[0])
        self.memories["eyebrow_layer"] = createMemory(self.decomposer.outputs[1])
        self.memories["image_prepared"] = createMemory(self.decomposer.outputs[2])

        self.memories['eyebrow_pose'] = createMemory(self.combiner.inputs[3])
        self.memories['eyebrow_image'] = createMemory(self.combiner.outputs[0])
        self.memories['morpher_decoded'] = createMemory(self.combiner.outputs[1])

        self.memories['face_pose'] = createMemory(self.morpher.inputs[2])
        self.memories['face_morphed_full'] = createMemory(self.morpher.outputs[0])
        self.memories['face_morphed_half'] = createMemory(self.morpher.outputs[1])

        self.memories['rotation_pose'] = createMemory(self.rotator.inputs[1])
        self.memories['wrapped_image'] = createMemory(self.rotator.outputs[0])
        self.memories['grid_change'] = createMemory(self.rotator.outputs[1])

        self.memories['output_img'] = createMemory(self.editor.outputs[0])
        self.memories['output_cv_img'] = createMemory(self.editor.outputs[1])

    def setMemsToEngines(self):
        """Connect allocated memory buffers to engine I/O ports"""
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Linking memories on VRAM to engine graph nodes')
        decomposer_inputs = [self.memories['input_img']]
        self.decomposer.setInputMems(decomposer_inputs)
        decomposer_outputs = [self.memories["background_layer"], self.memories["eyebrow_layer"], self.memories["image_prepared"]]
        self.decomposer.setOutputMems(decomposer_outputs)

        combiner_inputs = [self.memories['image_prepared'], self.memories["background_layer"], self.memories["eyebrow_layer"], self.memories['eyebrow_pose']]
        self.combiner.setInputMems(combiner_inputs)
        combiner_outputs = [self.memories['eyebrow_image'], self.memories['morpher_decoded']]
        self.combiner.setOutputMems(combiner_outputs)

        morpher_inputs = [self.memories['image_prepared'], self.memories['eyebrow_image'], self.memories['face_pose'], self.memories['morpher_decoded']]
        self.morpher.setInputMems(morpher_inputs)
        morpher_outputs = [self.memories['face_morphed_full'], self.memories['face_morphed_half']]
        self.morpher.setOutputMems(morpher_outputs)

        rotator_inputs = [self.memories['face_morphed_half'], self.memories['rotation_pose']]
        self.rotator.setInputMems(rotator_inputs)
        rotator_outputs = [self.memories['wrapped_image'], self.memories['grid_change']]
        self.rotator.setOutputMems(rotator_outputs)

        editor_inputs = [self.memories['face_morphed_full'], self.memories['wrapped_image'], self.memories['grid_change'], self.memories['rotation_pose']]
        self.editor.setInputMems(editor_inputs)
        editor_outputs = [self.memories['output_img'], self.memories['output_cv_img']]
        self.editor.setOutputMems(editor_outputs)
