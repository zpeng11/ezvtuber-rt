from ezvtb_rt.trt_utils import *
from ezvtb_rt.rife import RIFE
from ezvtb_rt.tha import THA
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr import SR
from ezvtb_rt.common import Core

class CoreTRT(Core):
    """Main inference pipeline combining THA face model with optional components:
    - RIFE for frame interpolation
    - SR for super resolution
    - Cacher for output caching
    
    Args:
        tha_dir: Path to THA model directory
        vram_cache_size: VRAM allocated for model caching (GB)
        use_eyebrow: Enable eyebrow motion processing
        rife_dir: Path to RIFE model directory (None to disable)
        sr_dir: Path to SR model directory (None to disable) 
        cache_max_volume: Max disk cache size (GB)
        cache_quality: Cache compression quality (1-3)
    """
    def __init__(self, tha_dir:str, vram_cache_size:float, use_eyebrow:bool, rife_dir:str, sr_dir:str,  cache_max_volume:float, cache_quality:int = 2):
        # Initialize core THA face model
        self.tha = THA(tha_dir, vram_cache_size, use_eyebrow)

        # Initialize optional components
        self.rife = None  # Frame interpolation module
        self.sr = None    # Super resolution module
        self.cacher = None# Output caching system
        self.scale = 1    # Output scaling factor

        # Initialize RIFE if model path provided
        if rife_dir is not None:
            self.rife = RIFE(rife_dir, self.tha.instream, self.tha.memories['output_cv_img'])
            self.scale = self.rife.scale
        # Initialize SR if model path provided
        if sr_dir is not None:
            instream = None  # Will be set based on RIFE/THA
            mems = []        # Memory buffers from previous stage
            if self.rife is not None:
                instream = self.rife.instream
                for i in range(self.rife.scale):
                    mems.append(self.rife.memories['framegen_'+str(i)])
            else:
                instream = self.tha.instream
                mems.append(self.tha.memories['output_cv_img'])
            self.sr = SR(sr_dir, instream, mems)

        # Initialize cache if enabled
        if cache_max_volume > 0.0:
            self.cacher = Cacher(cache_max_volume, cache_quality)

    def setImage(self, img:np.ndarray):
        """Set input image for processing pipeline
        Args:
            img: Input image in BGR format (HWC, uint8)
        """
        self.tha.setImage(img)

    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        """Run full inference pipeline
        Args:
            pose: Facial pose parameters (45 floats)
            
        Returns:
            List of output images from final stage in pipeline.
            Note: Numpy arrays should be copied/used before next inference
        """
        # Convert pose to required precision
        pose = pose.astype(np.float32)

        # Cache management variables
        need_cache_write = 0  # Hash value if cache needs updating
        res_carrier = None    # Current result container

        # Cache bypass path
        if self.cacher is None:
            # Directly run THA inference
            self.tha.inference(pose)
            res_carrier = self.tha
        else:  # Cache enabled path
            hs = hash(str(pose))  # Create pose hash key
            cached = self.cacher.read(hs)

            if cached is not None:  # Cache hit
                # Copy cached data to GPU memory
                np.copyto(self.tha.memories['output_cv_img'].host, cached)
                self.tha.memories['output_cv_img'].htod(self.tha.instream)
                res_carrier = [cached]
            else:  # Cache miss
                # Run THA inference and flag for cache storage
                self.tha.inference(pose)
                need_cache_write = hs
                res_carrier = self.tha

        # Run frame interpolation if enabled
        if self.rife is not None:
            self.rife.inference()
            res_carrier = self.rife
        # Run super resolution if enabled
        if self.sr is not None:
            self.sr.inference()
            res_carrier = self.sr

        # Update cache if we had a miss
        if need_cache_write != 0:
            self.cacher.write(need_cache_write, self.tha.fetchRes()[0])

        if type(res_carrier) is not list:
            res_carrier = res_carrier.fetchRes()
        return res_carrier
