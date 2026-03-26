from typing import List, Optional
import cv2
import numpy as np
import os
from ezvtb_rt.tha3_ort import THA3ORTSessions, THA3ORTNonDefaultSessions
from ezvtb_rt.cache import Cacher
from ezvtb_rt.tha4_ort import THA4ORTSessions, THA4ORTNonDefaultSessions
from ezvtb_rt.tha4_student_ort import THA4StudentORTSessions
import ezvtb_rt
from ezvtb_rt.ort_utils import createORTSession
import pyanime4k


def _mark_interpolated_frames(frames: np.ndarray) -> None:
    """插值帧打红点、原始帧打蓝点，仅当环境变量 EZVTB_MARK_INTERPOLATED=1 时生效。
    支持 NHWC (N,H,W,4) 与 NCHW (N,4,H,W)，ORT 多为 NHWC。"""
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
        # NCHW，4x4 方块
        for i in range(n - 1):
            frames[i, :, 0:4, 0:4] = np.array(red_bgra, dtype=frames.dtype).reshape(4, 1, 1)
        frames[n - 1, :, 0:4, 0:4] = np.array(blue_bgra, dtype=frames.dtype).reshape(4, 1, 1)
    else:
        # NHWC，4x4 方块
        for i in range(n - 1):
            frames[i, 0:4, 0:4, :] = red_bgra
        frames[n - 1, 0:4, 0:4, :] = blue_bgra


class CoreORT:
    def __init__(self,
                 tha_model_version: str = 'v3',
                 tha_model_seperable: bool = True,
                 tha_model_fp16: bool = False,
                 tha_model_name: str = None,
                 rife_model_enable: bool = False,
                 rife_model_scale: int = 2,
                 rife_model_fp16: bool = False,
                 sr_model_enable: bool = False,
                 sr_model_scale: int = 2,
                 sr_model_fp16: bool = False,
                 sr_a4k: bool = False,
                 vram_cache_size: float = 1.0,
                 cache_max_giga: float = 2.0,
                 use_eyebrow: bool = False):
        if tha_model_version == 'v3':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha3',
                                    'seperable' if tha_model_seperable else 'standard',
                                    'fp16' if tha_model_fp16 else 'fp32')
        elif tha_model_version == 'v4':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha4',
                                    'fp16' if tha_model_fp16 else 'fp32')
        elif tha_model_version == 'v4_student':
            # Support custom student models in data/models/custom_tha4_models
            if tha_model_name:
                tha_path = os.path.normpath(os.path.join(
                    ezvtb_rt.EZVTB_DATA,
                    'custom_tha4_models', tha_model_name
                ))
            else:
                tha_path = os.path.join(
                    ezvtb_rt.EZVTB_DATA, 'tha4_student'
                )
        else:
            raise ValueError('Unsupported THA model version')

        sr_path = None
        if sr_model_enable and not sr_a4k:
            if sr_model_scale == 4:
                sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN',
                                       f'exported_256_{"fp16" if sr_model_fp16 else "fp32"}.onnx')
            else:  # x2
                sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x',
                                       f"noise0_scale2x_{'fp16' if sr_model_fp16 else 'fp32'}.onnx")

        device_id = int(os.environ.get('EZVTB_DEVICE_ID', '0'))
        if tha_model_version == 'v3':
            if device_id == 0:
                self.tha = THA3ORTSessions(tha_path, use_eyebrow)
            else:
                self.tha = THA3ORTNonDefaultSessions(tha_path, device_id, use_eyebrow)
        elif tha_model_version == 'v4_student':
            self.tha = THA4StudentORTSessions(tha_path, device_id)
        elif tha_model_version == 'v4':
            if device_id == 0:
                self.tha = THA4ORTSessions(tha_path, use_eyebrow)
            else:
                self.tha = THA4ORTNonDefaultSessions(tha_path, device_id, use_eyebrow)
        else:
            raise ValueError('Unsupported THA model version')
        self.tha_model_fp16: bool = tha_model_fp16
        self.v3: bool = (tha_model_version == 'v3')
        self.rife: Optional[ort.InferenceSession] = None
        self.smaller_rifes: List[ort.InferenceSession] = []
        self.rife_model_scale: int = rife_model_scale
        self.sr: Optional[ort.InferenceSession] = None
        self.sr_cacher: Optional[Cacher] = None
        self.sr_a4k = pyanime4k.Processor(
            processor_type="opencl",
            device=0,
            model="acnet-gan"
        ) if sr_a4k else None
        self.cacher: Optional[Cacher] = None
        self.last_tha_output: Optional[np.ndarray] = None

        if rife_model_enable:
            rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife',
                                     f'rife_x{rife_model_scale}_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
            self.rife = createORTSession(rife_path, device_id)
            if rife_model_scale >= 3:
                x2_rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife',
                                            f'rife_x2_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
                x2_rife = createORTSession(x2_rife_path, device_id)
                self.smaller_rifes.append(x2_rife)
            if rife_model_scale == 4:
                x3_rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife',
                                            f'rife_x3_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
                x3_rife = createORTSession(x3_rife_path, device_id)
                self.smaller_rifes.append(x3_rife)
        if sr_path is not None:
            self.sr = createORTSession(sr_path, device_id)
        if cache_max_giga > 0.0 and sr_model_enable:
            self.sr_cacher = Cacher(cache_max_giga, width=1024, height=1024)
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)

    def setImage(self, img: np.ndarray):
        self.tha.update_image(img)
        self.last_tha_output = img

    def inference(self, poses: List[np.ndarray]) -> np.ndarray:
        for i in range(len(poses)):
            poses[i] = poses[i].astype(np.float32)
            if self.tha_model_fp16 and not self.v3:  # For THA4 with FP16 model poses are fp16 inputs
                poses[i] = poses[i].astype(np.float16)

        tha_result: np.ndarray = self.cacher.get(hash(str(poses[-1]))) if self.cacher is not None else None
        if tha_result is None:  # Do not use cacher or cache missed
            tha_result = self.tha.inference(poses[-1])
            if self.cacher is not None:
                self.cacher.put(hash(str(poses[-1])), tha_result)

        if self.rife is None and self.sr is None and self.sr_a4k is None:  # Only THA
            return np.expand_dims(tha_result, axis=0)

        rife_result: np.ndarray = None
        if self.rife is not None:  # RIFE
            if len(poses) == 1:
                rife_result = self.rife.run(None, {'tha_img_0': np.expand_dims(self.last_tha_output, axis=0),
                                                   'tha_img_1': np.expand_dims(tha_result, axis=0)})[0]
            else:  # Multiple poses provided
                if self.cacher:
                    cached_rife = [self.cacher.get(hash(str(p))) for p in poses[:-1]]
                else:
                    cached_rife = [None] * (len(poses) - 1)
                if all(x is None for x in cached_rife):  # No cached frames
                    rife_result = self.rife.run(None, {'tha_img_0': np.expand_dims(self.last_tha_output, axis=0),
                                                       'tha_img_1': np.expand_dims(tha_result, axis=0)})[0]
                elif all(x is not None for x in cached_rife):  # All cached frames
                    # print('RIFE all frames cached')
                    rife_result = np.stack(cached_rife + [tha_result], axis=0)
                elif self.rife_model_scale == 3:
                    # One frame is not cached
                    rife_x2 = self.smaller_rifes[0]
                    if cached_rife[0] is None:
                        rife_x2_result = rife_x2.run(None, {'tha_img_0': np.expand_dims(self.last_tha_output, axis=0),
                                                            'tha_img_1': np.expand_dims(cached_rife[1], axis=0)})[0]
                        cached_rife[0] = rife_x2_result[0]
                    else:
                        rife_x2_result = rife_x2.run(None, {'tha_img_0': np.expand_dims(cached_rife[0], axis=0),
                                                            'tha_img_1': np.expand_dims(tha_result, axis=0)})[0]
                        cached_rife[1] = rife_x2_result[0]
                    rife_result = np.stack(cached_rife + [tha_result], axis=0)
                elif self.rife_model_scale == 4:
                    # One or two frames are not cached
                    rife_x2 = self.smaller_rifes[0]
                    rife_x3 = self.smaller_rifes[1]
                    cached_rife = [self.last_tha_output] + cached_rife + [tha_result]
                    number_of_missing = sum(1 for x in cached_rife if x is None)
                    if number_of_missing == 1:  # Only one frame is not cached
                        # print('RIFE x4 one frame cache miss')
                        missing_index = -1
                        for i in range(len(cached_rife)):
                            if cached_rife[i] is None:
                                missing_index = i
                                break
                        rife_x2_result = rife_x2.run(None, {'tha_img_0': np.expand_dims(cached_rife[i - 1], axis=0),
                                                            'tha_img_1': np.expand_dims(cached_rife[i + 1], axis=0)})[0]
                        cached_rife[missing_index] = rife_x2_result[0]
                        rife_result = np.stack(cached_rife[1:], axis=0)
                    elif number_of_missing == 2:
                        # print('RIFE x4 two frames cache miss')
                        if cached_rife[2] is not None:
                            cached_rife[1] = rife_x2.run(None, {'tha_img_0': np.expand_dims(cached_rife[0], axis=0),
                                                                'tha_img_1': np.expand_dims(cached_rife[2], axis=0)})[
                                0][0]
                            cached_rife[3] = rife_x2.run(None, {'tha_img_0': np.expand_dims(cached_rife[2], axis=0),
                                                                'tha_img_1': np.expand_dims(cached_rife[4], axis=0)})[
                                0][0]
                        elif cached_rife[1] is not None:
                            rife_x3_result = rife_x3.run(None, {'tha_img_0': np.expand_dims(cached_rife[1], axis=0),
                                                                'tha_img_1': np.expand_dims(cached_rife[4], axis=0)})[0]
                            cached_rife[2] = rife_x3_result[0]
                            cached_rife[3] = rife_x3_result[1]
                        else:  # cached_rife[3] is not None
                            rife_x3_result = rife_x3.run(None, {'tha_img_0': np.expand_dims(cached_rife[0], axis=0),
                                                                'tha_img_1': np.expand_dims(cached_rife[3], axis=0)})[0]
                            cached_rife[1] = rife_x3_result[0]
                            cached_rife[2] = rife_x3_result[1]
                        rife_result = np.stack(cached_rife[1:], axis=0)
                else:
                    raise ValueError('RIFE model scale not supported for partial caching')
                if self.cacher and len(poses) > 1:  # Update cache for newly computed frames
                    for i in range(len(poses) - 1):
                        self.cacher.put(hash(str(poses[i])), rife_result[i])
            self.last_tha_output = tha_result
        else:
            rife_result = np.expand_dims(tha_result, axis=0)

        if self.sr is None and self.sr_a4k is None:  # Only RIFE
            _mark_interpolated_frames(rife_result)
            return rife_result

        sr_process_func = self.sr_a4k_process if self.sr_a4k is not None else self.sr_onnx_process
        # SR
        if len(poses) == 1:  # Only one pose provided,
            hs = hash(str(poses[-1]))
            cached_sr = self.sr_cacher.get(hs) if self.sr_cacher is not None else None
            sr_batch = rife_result.shape[0]
            if sr_batch > 1:  # Multiple frames
                if cached_sr is None:  # No cached frame
                    sr_result = sr_process_func(rife_result)
                    if self.sr_cacher is not None:
                        self.sr_cacher.put(hs, sr_result[-1])
                else:  # The last frame is cached
                    sr_result = sr_process_func(rife_result[:sr_batch - 1])
                    sr_result = np.concatenate([sr_result, np.expand_dims(cached_sr, axis=0)], axis=0)
            else:  # The only frame
                if cached_sr is not None:  # Cached
                    sr_result = np.expand_dims(cached_sr, axis=0)
                else:  # Not cached
                    sr_result = sr_process_func(rife_result)
                    if self.sr_cacher is not None:
                        self.sr_cacher.put(hs, sr_result[-1])
        else:  # Multiple poses provided for multiple frames
            assert len(poses) == rife_result.shape[0]
            all_cached: bool = self.sr_cacher is not None and all(
                self.sr_cacher.query(hash(str(pose))) for pose in poses)
            if all_cached:  # All frames are cached, this is a quick path
                sr_result = np.stack([self.sr_cacher.get(hash(str(pose))) for pose in poses], axis=0)
            else:  # Some frames are not cached
                if self.sr_cacher is None:  # No cacher, process all frames directly
                    sr_result = sr_process_func(rife_result)
                else:  # With cacher, check each frame
                    sr_result = []
                    to_sr_images = []
                    for i in range(len(poses)):
                        hs = hash(str(poses[i]))
                        cached_sr = self.sr_cacher.get(hs)
                        if cached_sr is None:
                            to_sr_images.append(rife_result[i])
                            sr_result.append(None)  # Placeholder
                        else:
                            sr_result.append(cached_sr)
                    assert len(to_sr_images) > 0
                    to_sr_images_np = np.stack(to_sr_images, axis=0)
                    sr_outputs = sr_process_func(to_sr_images_np)
                    sr_idx = 0
                    for i in range(len(poses)):
                        if sr_result[i] is None:
                            sr_result[i] = sr_outputs[sr_idx]
                            hs = hash(str(poses[i]))
                            self.sr_cacher.put(hs, sr_outputs[sr_idx])
                            sr_idx += 1
                    sr_result = np.stack(sr_result, axis=0)
        _mark_interpolated_frames(sr_result)
        return sr_result

    def sr_onnx_process(self, frames: np.ndarray) -> np.ndarray:
        assert self.sr is not None, "SR ONNX model is not initialized."
        assert len(frames.shape) == 4, "Input frames should have 4 dimensions (batch, height, width, channels)."
        sr_result = self.sr.run(None, {self.sr.get_inputs()[0].name: frames})[0]
        return sr_result

    def sr_a4k_process(self, frames: np.ndarray) -> np.ndarray:
        assert self.sr_a4k is not None, "Anime4K processor is not initialized."
        assert len(frames.shape) == 4, "Input frames should have 4 dimensions (batch, height, width, channels)."
        sr_result = []
        for i in range(frames.shape[0]):
            alpha_channel = np.ascontiguousarray(frames[i, :, :, 3])  # Preserve alpha channel
            rgb_channels = np.ascontiguousarray(frames[i, :, :, :3])  # RGB channels
            processed_rgb = self.sr_a4k.process(rgb_channels)
            processed_alpha = cv2.resize(alpha_channel, None, fx=2, fy=2)
            rgba_image = cv2.merge((processed_rgb, processed_alpha))
            sr_result.append(rgba_image)
        sr_result = np.stack(sr_result, axis=0)
        return sr_result
