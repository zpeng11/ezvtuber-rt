import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import check_build_all_models
from ezvtb_rt.core import Core
from ezvtb_rt.cache import Cacher
from ezvtb_rt.rife import RIFECoreLinked
from ezvtb_rt.tha import THACoreCachedRAM, THACoreCachedVRAM
from ezvtb_rt.sr import SRLinkRife, SRLinkTha
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import generate_video
import json
import cv2


def CorePerf():
    
    tha_core = THACoreCachedVRAM('./data/tha3/standard/fp16')
    rife_core = RIFECoreLinked(f'./data/rife_512/x4/fp16', tha_core)

    # sr_core = SRLinkTha('data\\Real-ESRGAN\\exported_256_fp16', tha_core)
    sr_core = SRLinkTha('data\\waifu2x_upconv\\fp16\\upconv_7\\photo\\noise2_scale2x', tha_core)
    
    cacher = Cacher(cache_quality=90, image_size=1024, max_size=10)
    core = Core(tha_core, cacher, sr_core, None)
    core.setImage( cv2.imread('f:/talking-head-anime-3-demo/data/images/crypko_01.png', cv2.IMREAD_UNCHANGED))
    cuda.start_profiler()
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for i in tqdm(range(len(pose_data))):
        ret = core.inference(np.array(pose_data[i]).reshape(1,45))
        for item in ret:
            item.copy()
    for i in tqdm(range(len(pose_data))):
        ret = core.inference(np.array(pose_data[i]).reshape(1,45))
        for item in ret:
            item.copy()
    cuda.stop_profiler()

def CoreShow():
    tha_core = THACoreCachedVRAM('./data/tha3/seperable/fp16')
    rife_core = RIFECoreLinked(f'./data/rife_512/x4/fp16', tha_core)

    sr_core = SRLinkTha('data\\Real-ESRGAN\\exported_256_fp16', tha_core)
    # sr_core = SRLinked('data\\waifu2x_upconv\\fp16\\upconv_7\\photo\\noise2_scale2x', tha_core)
    
    cacher = Cacher(cache_quality=90, image_size=512)
    core = Core(tha_core, cacher, None, rife_core)
    core.setImage( cv2.imread('f:/talking-head-anime-3-demo/data/images/crypko_01.png', cv2.IMREAD_UNCHANGED))

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    pose_data = pose_data[800:1000]

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output.copy())
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output.copy())
        return new_vid
    cuda.start_profiler()
    new_vid = createInterpolatedVideo(pose_data, core)
    cuda.stop_profiler()
    for i in range(len(new_vid)):
        new_vid[i] = new_vid[i][:,:,:3] 
    generate_video(new_vid, './test/data/core/test.mp4', 80)
    if core.cacher is not None:
        print(core.cacher.hits, core.cacher.miss)


if __name__ == "__main__":
    # check_build_all_models()
    CoreShow()
    # CorePerf()