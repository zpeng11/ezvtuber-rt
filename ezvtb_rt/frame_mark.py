import numpy as np
import os

def mark_interpolated_frames(frames: np.ndarray) -> bool:
    """插值帧打红点、原始帧打蓝点，仅当环境变量 EZVTB_MARK_INTERPOLATED=1 时生效。
    假设 N=-1 是原始帧，0~N-2 是插值帧，则在每帧左上角 4x4 区域打红点或蓝点。
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