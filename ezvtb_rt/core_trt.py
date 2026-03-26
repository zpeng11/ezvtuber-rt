from ezvtb_rt.trt_utils import *
from ezvtb_rt.trt_engine import TRTEngine, HostDeviceMem
from ezvtb_rt.tha3 import THA3Engines
from ezvtb_rt.tha4 import THA4Engines
from ezvtb_rt.tha4_student import THA4StudentEngines
from ezvtb_rt.cache import Cacher
import ezvtb_rt
import numpy as np
import os
from typing import List
import pyanime4k
import cv2


def _mark_interpolated_frames(frames: np.ndarray) -> None:
    """插值帧打红点、原始帧打蓝点，仅当环境变量 EZVTB_MARK_INTERPOLATED=1 时生效。
    支持 NHWC (N,H,W,4) 与 NCHW (N,4,H,W)，TRT 多为 NCHW，ORT 多为 NHWC。"""
    if not os.environ.get('EZVTB_MARK_INTERPOLATED', '').strip() in ('1', 'true', 'True', 'yes'):
        return
    if len(frames.shape) != 4 or frames.shape[0] < 1:
        return
    if frames.dtype == np.uint8:
        red_bgra = (0, 0, 255, 255)
        blue_bgra = (255, 0, 0, 255)
    else:
        red_bgra = (0.0, 0.0, 1.0, 1.0)
        blue_bgra = (1.0, 0.0, 0.0, 1.0)
    n = frames.shape[0]
    if frames.shape[1] == 4:
        # NCHW (TensorRT 常见)，4x4 方块
        for i in range(n - 1):
            frames[i, :, 0:4, 0:4] = np.array(red_bgra, dtype=frames.dtype).reshape(4, 1, 1)
        frames[n - 1, :, 0:4, 0:4] = np.array(blue_bgra, dtype=frames.dtype).reshape(4, 1, 1)
    else:
        # NHWC (ONNX 常见)，4x4 方块
        for i in range(n - 1):
            frames[i, 0:4, 0:4, :] = red_bgra
        frames[n - 1, 0:4, 0:4, :] = blue_bgra


def has_none_object_none_pattern(lst):
    if len(lst) < 3:
        return False
    for i in range(len(lst) - 2):
        if lst[i] is None and lst[i+1] is not None and lst[i+2] is None:
            return True
    return False

def find_none_block_indices(lst):
    if not lst:
        return None, None  # Empty list: no indices
    
    none_indices = [i for i, x in enumerate(lst) if x is None]
    if not none_indices:
        return None, None  # No Nones: no block
    
    first_none = min(none_indices)
    last_none = max(none_indices)
    
    # Optional: Verify contiguous block (all between first and last are None)
    if any(lst[i] is not None for i in range(first_none, last_none + 1)):
        raise ValueError("Nones are not contiguous")
    
    i1 = first_none - 1 if first_none > 0 else None  # None if block starts at 0
    i2 = last_none + 1 if last_none < len(lst) - 1 else None  # None if block ends at last index
    
    return i1, i2

class CoreTRT:
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
        cache_max_giga: Max disk cache size (GB)
    """
    def __init__(self, 
                 tha_model_version:str = 'v3',
                 tha_model_seperable:bool = True,
                 tha_model_fp16:bool = False,
                 tha_model_name:str = None,
                 rife_model_enable:bool = False,
                 rife_model_scale:int = 2,
                 rife_model_fp16:bool = False,
                 sr_model_enable:bool = False,
                 sr_model_scale:int = 2,
                 sr_model_fp16:bool = False,
                 sr_a4k:bool = False,
                 vram_cache_size:float = 1.0, 
                 cache_max_giga:float = 2.0, 
                 use_eyebrow:bool = False):
        if tha_model_version == 'v3':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha3',
                                    'seperable' if tha_model_seperable else 'standard', 
                                    'fp16' if tha_model_fp16 else 'fp32')
            self.v3 = True
        elif tha_model_version == 'v4':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha4', 
                                    'fp16' if tha_model_fp16 else 'fp32')
            self.v3 = False
        elif tha_model_version == 'v4_student':
            # Support custom student models in data/models/custom_tha4_models
            if tha_model_name:
                # Build path relative to project root (parent of ezvtuber-rt)
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(__file__))
                )
                tha_path = os.path.normpath(os.path.join(
                    project_root, 'data', 'models',
                    'custom_tha4_models', tha_model_name
                ))
            else:
                tha_path = os.path.join(
                    ezvtb_rt.EZVTB_DATA, 'tha4_student'
                )
            self.v3 = False
        else:
            raise ValueError('Unsupported THA model version')
            
        sr_path = None
        if sr_model_enable and not sr_a4k:
            if sr_model_scale == 4:
                if sr_model_fp16:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', 'exported_256_fp16.onnx')
                else:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', 'exported_256_fp32.onnx')
            else: #x2
                if sr_model_fp16:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x', 'noise0_scale2x_fp16.onnx')
                else:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x', 'noise0_scale2x_fp32.onnx')

        # Initialize core THA face model
        if self.v3:
            self.tha = THA3Engines(tha_path, vram_cache_size, use_eyebrow)
        elif tha_model_version == 'v4_student':
            self.tha = THA4StudentEngines(tha_path)
        elif tha_model_version == 'v4':
            self.tha = THA4Engines(tha_path, vram_cache_size, use_eyebrow)
        else:
            raise ValueError('Unsupported THA model version')

        self.tha_model_fp16: bool = tha_model_fp16

        # Initialize optional components
        self.rife: TRTEngine = None  # Frame interpolation module (default)
        self.smaller_rifes: List[TRTEngine] = []  # Additional RIFE engines for different scales
        self.rife_model_scale: int = rife_model_scale
        self.sr: TRTEngine = None    # Super resolution module
        self.cacher: Cacher = None# Output caching system
        self.sr_cacher: Cacher = None # SR output caching
        self.sr_a4k = pyanime4k.Processor(
                processor_type="opencl",
                device=0,
                model="acnet-gan"
            ) if sr_a4k else None
        self.last_tha_output: np.ndarray | None = None

        # Initialize RIFE if model path provided
        if rife_model_enable:
            rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife', 
                                     f'rife_x{rife_model_scale}_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
            self.rife = TRTEngine(rife_path, 2)
            self.rife.configure_in_out_tensors()
            if rife_model_scale >= 3:
                x2_rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife',
                                            f'rife_x2_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
                x2_rife = TRTEngine(x2_rife_path, 2)
                x2_rife.configure_in_out_tensors()
                self.smaller_rifes.append(x2_rife)
            if rife_model_scale == 4:
                x3_rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife', 
                                            f'rife_x3_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
                x3_rife = TRTEngine(x3_rife_path, 2)
                x3_rife.configure_in_out_tensors()
                self.smaller_rifes.append(x3_rife)

        # Initialize SR if model path provided
        if sr_path is not None:
            self.sr = TRTEngine(sr_path, 1)
            self.sr.configure_in_out_tensors(rife_model_scale if rife_model_enable else 1)
        if cache_max_giga > 0.0 and sr_model_enable:
            # SR outputs are upscaled (expected 1024x1024 RGBA)
            self.sr_cacher = Cacher(cache_max_giga, width=1024, height=1024)

        # Initialize cache if enabled
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)

        self.main_stream: cuda.Stream = cuda.Stream()
        self.cache_stream: cuda.Stream = cuda.Stream()

    def setImage(self, img:np.ndarray):
        """Set input image for processing pipeline
        Args:
            img: Input image in BGR format (HWC, uint8)
        """
        self.tha.syncSetImage(img)
        self.last_tha_output = img

    def inference(self, poses: List[np.ndarray]) -> np.ndarray:
        """Run full inference pipeline
        Args:
            poses: One or more facial pose arrays. If multiple are provided,
                   RIFE interpolation and per-pose caching are enabled similar to ORT.
        Returns:
            Batched output images from the final stage in the pipeline.
        """
        assert isinstance(poses, list) and all(isinstance(p, np.ndarray) for p in poses), "poses must be a list of numpy arrays"

        if len(poses) == 0:
            raise ValueError('poses must not be empty')

        # Normalize dtype for all poses
        for i in range(len(poses)):
            poses[i] = poses[i].astype(np.float32)
            if self.tha_model_fp16 and not self.v3:
                poses[i] = poses[i].astype(np.float16)

        tha_pose = poses[-1]

        tha_mem_res: HostDeviceMem = self.tha.getOutputMem()

        cached_output = None
        # THA cache lookup for the last pose only (matches ORT semantics)
        if self.cacher is not None:
            self.cache_stream.synchronize()
            cached_output = self.cacher.get(hash(str(tha_pose)))
            if cached_output is not None:
                np.copyto(tha_mem_res.host, cached_output)
                tha_mem_res.htod(self.main_stream)
        # Run THA when not cached
        if cached_output is None:
            self.tha.asyncInfer(tha_pose, self.main_stream)
            # Need host data for caching or SR-only path
            tha_mem_res.dtoh(self.main_stream)
            self.main_stream.synchronize()
            if self.cacher is not None:
                self.cacher.put(hash(str(tha_pose)), tha_mem_res.host)

        # If no RIFE and no SR, just return THA result
        if self.rife is None and self.sr is None and self.sr_a4k is None:
            return np.expand_dims(cached_output if cached_output is not None else np.copy(tha_mem_res.host), axis=0)
        
        # RIFE interpolation stage
        rife_mem_res : HostDeviceMem = None
        if self.rife is not None:
            rife_mem_res = self.rife.outputs[0]
            if len(poses) == 1:
                # Prepare previous frame
                np.copyto(self.rife.inputs[0].host, np.expand_dims(self.last_tha_output, axis=0))
                self.rife.inputs[0].htod(self.main_stream)
                # Current frame
                self.rife.inputs[1].bridgeFrom(tha_mem_res, self.main_stream)
                self.rife.asyncKickoff(self.main_stream)
                self.rife.outputs[0].dtoh(self.main_stream)
                self.main_stream.synchronize()
            else:
                if self.cacher is not None:
                    cached_rife = [self.cacher.get(hash(str(p))) for p in poses[:-1]]
                else:
                    cached_rife = [None] * (len(poses) -1)
                if all(x is None for x in cached_rife): # No cached frames
                    # Prepare previous frame
                    np.copyto(self.rife.inputs[0].host, np.expand_dims(self.last_tha_output, axis=0))
                    self.rife.inputs[0].htod(self.main_stream)
                    # Current frame
                    self.rife.inputs[1].bridgeFrom(tha_mem_res, self.main_stream)
                    self.rife.asyncKickoff(self.main_stream)
                    self.rife.outputs[0].dtoh(self.main_stream)
                    self.main_stream.synchronize()
                elif all(x is not None for x in cached_rife): # All cached frames
                    # print('RIFE all frames cached')
                    rife_result = np.stack(cached_rife + [tha_mem_res.host], axis=0)
                    np.copyto(rife_mem_res.host, rife_result)
                    rife_mem_res.htod(self.main_stream)
                    self.main_stream.synchronize()
                elif self.rife_model_scale == 3:
                    # rife x3 one frame missing
                    # print('RIFE x3 one frame cache miss')
                    rife_2x = self.smaller_rifes[0]
                    if cached_rife[0] is None:
                        # First frame missing
                        np.copyto(rife_2x.inputs[0].host, np.expand_dims(self.last_tha_output, axis=0))
                        rife_2x.inputs[0].htod(self.main_stream)
                        np.copyto(rife_2x.inputs[1].host, np.expand_dims(cached_rife[1], axis=0))
                        rife_2x.inputs[1].htod(self.main_stream)
                        rife_2x.asyncKickoff(self.main_stream)
                        rife_2x.outputs[0].dtoh(self.main_stream)
                        self.main_stream.synchronize()
                        cached_rife[0] = rife_2x.outputs[0].host[0]
                    else: # cached_rife[1] is None
                        np.copyto(rife_2x.inputs[0].host, np.expand_dims(cached_rife[0], axis=0))
                        rife_2x.inputs[0].htod(self.main_stream)
                        rife_2x.inputs[1].bridgeFrom(tha_mem_res, self.main_stream)
                        rife_2x.asyncKickoff(self.main_stream)
                        rife_2x.outputs[0].dtoh(self.main_stream)
                        self.main_stream.synchronize()
                        cached_rife[1] = rife_2x.outputs[0].host[0]
                    rife_result = np.stack(cached_rife + [tha_mem_res.host], axis=0)
                    np.copyto(rife_mem_res.host, rife_result)
                    rife_mem_res.htod(self.main_stream)
                    self.main_stream.synchronize()
                elif self.rife_model_scale == 4:
                    # One or two frames missing with rife x4
                    rife_x2 = self.smaller_rifes[0]
                    rife_x3 = self.smaller_rifes[1]
                    cached_rife = [self.last_tha_output] + cached_rife + [tha_mem_res.host]
                    number_of_missing = sum(1 for x in cached_rife if x is None)
                    if number_of_missing == 1: # Only one frame is not cached
                        # print('RIFE x4 one frame cache miss')
                        missing_index = -1
                        for i in range(len(cached_rife)):
                            if cached_rife[i] is None:
                                missing_index = i
                                break
                        np.copyto(rife_x2.inputs[0].host, np.expand_dims(cached_rife[missing_index -1], axis=0))
                        rife_x2.inputs[0].htod(self.main_stream)
                        np.copyto(rife_x2.inputs[1].host, np.expand_dims(cached_rife[missing_index +1], axis=0))
                        rife_x2.inputs[1].htod(self.main_stream)
                        rife_x2.asyncKickoff(self.main_stream)
                        rife_x2.outputs[0].dtoh(self.main_stream)
                        self.main_stream.synchronize()
                        cached_rife[missing_index] = rife_x2.outputs[0].host[0]
                    elif number_of_missing == 2:
                        # print('RIFE x4 two frames cache miss')
                        if cached_rife[2] is not None:
                            np.copyto(rife_x2.inputs[0].host, np.expand_dims(cached_rife[0], axis=0))
                            rife_x2.inputs[0].htod(self.main_stream)
                            np.copyto(rife_x2.inputs[1].host, np.expand_dims(cached_rife[2], axis=0))
                            rife_x2.inputs[1].htod(self.main_stream)
                            rife_x2.asyncKickoff(self.main_stream)
                            rife_x2.outputs[0].dtoh(self.main_stream)
                            self.main_stream.synchronize()
                            cached_rife[1] = rife_x2.outputs[0].host[0]
                            np.copyto(rife_x2.inputs[0].host, np.expand_dims(cached_rife[2], axis=0))
                            rife_x2.inputs[0].htod(self.main_stream)
                            np.copyto(rife_x2.inputs[1].host, np.expand_dims(cached_rife[4], axis=0))
                            rife_x2.inputs[1].htod(self.main_stream)
                            rife_x2.asyncKickoff(self.main_stream)
                            rife_x2.outputs[0].dtoh(self.main_stream)
                            self.main_stream.synchronize()
                            cached_rife[3] = rife_x2.outputs[0].host[0]
                        elif cached_rife[1] is not None:
                            np.copyto(rife_x3.inputs[0].host, np.expand_dims(cached_rife[1], axis=0))
                            rife_x3.inputs[0].htod(self.main_stream)
                            np.copyto(rife_x3.inputs[1].host, np.expand_dims(cached_rife[4], axis=0))
                            rife_x3.inputs[1].htod(self.main_stream)
                            rife_x3.asyncKickoff(self.main_stream)
                            rife_x3.outputs[0].dtoh(self.main_stream)
                            self.main_stream.synchronize()
                            cached_rife[2] = rife_x3.outputs[0].host[0]
                            cached_rife[3] = rife_x3.outputs[0].host[1]
                        else: # cached_rife[3] is not None
                            np.copyto(rife_x3.inputs[0].host, np.expand_dims(cached_rife[0], axis=0))
                            rife_x3.inputs[0].htod(self.main_stream)
                            np.copyto(rife_x3.inputs[1].host, np.expand_dims(cached_rife[3], axis=0))
                            rife_x3.inputs[1].htod(self.main_stream)
                            rife_x3.asyncKickoff(self.main_stream)
                            rife_x3.outputs[0].dtoh(self.main_stream)
                            self.main_stream.synchronize()
                            cached_rife[1] = rife_x3.outputs[0].host[0]
                            cached_rife[2] = rife_x3.outputs[0].host[1]
                    else:
                        raise ValueError('RIFE x4 more than two missing frames not supported')
                    rife_result = np.stack(cached_rife[1:], axis=0)
                    np.copyto(rife_mem_res.host, rife_result)
                    rife_mem_res.htod(self.main_stream)
                    self.main_stream.synchronize()
                if self.cacher is not None:
                    for i in range(1, len(poses) -1):
                        self.cacher.put(hash(str(poses[i])), rife_mem_res.host[i])
            # Track last THA output for future interpolation
            self.last_tha_output = np.copy(tha_mem_res.host)
        else:
            # No RIFE, SR-only uses THA output as a single-frame batch
            rife_mem_res = tha_mem_res
        
        if self.sr is None and self.sr_a4k is None:
            _mark_interpolated_frames(rife_mem_res.host)
            return np.copy(rife_mem_res.host)
        
        if self.sr is not None:
            return self.sr_trt_process(poses, rife_mem_res)
        else: # self.sr_a4k is not None
            to_sr_images = rife_mem_res.host if len(rife_mem_res.host.shape) == 4 else np.expand_dims(rife_mem_res.host, axis=0)
            return self.sr_a4k_process(poses, to_sr_images)
    
    def sr_a4k_process(self, poses: List[np.ndarray], to_sr_images: np.ndarray) -> np.ndarray:
        sr_results = []
        if len(poses) == 1:
            for i in range(to_sr_images.shape[0] - 1):
                sr_results.append(self.a4k_infer_bgra(to_sr_images[i]))
            hs = hash(str(poses[0]))
            cached_sr = self.sr_cacher.get(hs) if self.sr_cacher is not None else None
            if cached_sr is not None:
                sr_results.append(cached_sr)
            else:
                sr_results.append(self.a4k_infer_bgra(to_sr_images[-1]))
                if self.sr_cacher is not None:
                    self.sr_cacher.put(hs, sr_results[-1])
        else:
            assert to_sr_images.shape[0] == len(poses)
            for i in range(len(poses)):
                hs = hash(str(poses[i]))
                cached_sr = self.sr_cacher.get(hs) if self.sr_cacher is not None else None
                if cached_sr is not None:
                    sr_results.append(cached_sr)
                else:
                    sr_img = self.a4k_infer_bgra(to_sr_images[i])
                    sr_results.append(sr_img)
                    if self.sr_cacher is not None:
                        self.sr_cacher.put(hs, sr_img)
        result = np.stack(sr_results, axis=0)
        _mark_interpolated_frames(result)
        return result

    def a4k_infer_bgra(self, img_bgra: np.ndarray) -> np.ndarray:
        assert self.sr_a4k is not None, "a4k_infer_bgra called but sr_a4k is not initialized"
        assert len(img_bgra.shape) == 3 and img_bgra.shape[2] == 4, "Input image must be BGRA format"
        alpha_channel = np.ascontiguousarray(img_bgra[:, :, 3:])  # Preserve alpha channel
        bgr_channels = np.ascontiguousarray(img_bgra[:, :, :3])
        sr_bgr = self.sr_a4k.process(bgr_channels)
        sr_alpha = cv2.resize(alpha_channel, None, fx=2, fy=2)
        sr_bgra = cv2.merge([sr_bgr, sr_alpha])
        return sr_bgra

    def sr_trt_process(self, poses: List[np.ndarray], rife_mem_res: HostDeviceMem)-> np.ndarray:
        # Special handling when only one pose was provided
        if len(poses) == 1:
            hs = hash(str(poses[0]))
            cached_sr = None if self.sr_cacher is None else self.sr_cacher.get(hs)
            if cached_sr is not None: # SR cache hit
                # Run SR on remaining frames only
                if len(rife_mem_res.host.shape) == 4 and rife_mem_res.host.shape[0] > 1:
                    res_host = rife_mem_res.host[:-1]
                    sr_batch = res_host.shape[0]
                    self.sr.configure_in_out_tensors(sr_batch)
                    cuda.memcpy_dtod_async(self.sr.inputs[0].device, rife_mem_res.device, res_host.nbytes, self.main_stream)
                    self.sr.asyncKickoff(self.main_stream)
                    self.sr.outputs[0].dtoh(self.main_stream)
                    self.main_stream.synchronize()
                    out = np.concatenate((self.sr.outputs[0].host, np.expand_dims(cached_sr, axis=0)), axis=0)
                    _mark_interpolated_frames(out)
                    return out
                else: # there is only one frame to SR, which is cached
                    out = np.expand_dims(cached_sr, axis=0)
                    _mark_interpolated_frames(out)
                    return out
            else: # No SR cache hit, dealing with single pose with RIFE interpolation followed by SR, or SR for a single tha result
                sr_batch = rife_mem_res.host.shape[0] if len(rife_mem_res.host.shape) == 4 else 1
                self.sr.configure_in_out_tensors(sr_batch)
                self.sr.inputs[0].bridgeFrom(rife_mem_res, self.main_stream)
                self.sr.asyncKickoff(self.main_stream)
                self.sr.outputs[0].dtoh(self.main_stream)
                self.main_stream.synchronize()
                if self.sr_cacher is not None:
                    self.sr_cacher.put(hs, self.sr.outputs[0].host[-1])
                out = np.copy(self.sr.outputs[0].host)
                _mark_interpolated_frames(out)
                return out
        else: # Multiple poses with RIFE followed by SR
            assert len(rife_mem_res.host.shape) == 4 and rife_mem_res.host.shape[0] == len(poses)
            sr_results = [None] * len(poses)
            to_sr_images = []
            for i in range(len(poses)):
                hs = hash(str(poses[i]))
                sr_results[i] = None if self.sr_cacher is None else self.sr_cacher.get(hs)
                if sr_results[i] is None:
                    to_sr_images.append(rife_mem_res.host[i])
            if all(x is not None for x in sr_results): # All SR cached
                out = np.stack(sr_results, axis=0)
                _mark_interpolated_frames(out)
                return out
            else: # Some SR missing, run SR on missing ones
                sr_batch = len(to_sr_images)
                self.sr.configure_in_out_tensors(sr_batch)
                for i, to_sr_image in enumerate(to_sr_images):
                    np.copyto(self.sr.inputs[0].host[i], to_sr_image)
                self.sr.inputs[0].htod(self.main_stream)
                self.sr.asyncKickoff(self.main_stream)
                self.sr.outputs[0].dtoh(self.main_stream)
                self.main_stream.synchronize()
                # Fill in SR results and update cache
                sr_output_idx = 0
                for i in range(len(poses)):
                    if sr_results[i] is None:
                        sr_results[i] = np.copy(self.sr.outputs[0].host[sr_output_idx])
                        if self.sr_cacher is not None:
                            self.sr_cacher.put(hash(str(poses[i])), sr_results[i])
                        sr_output_idx += 1
                out = np.stack(sr_results, axis=0)
                _mark_interpolated_frames(out)
                return out     
