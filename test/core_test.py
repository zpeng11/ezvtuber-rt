import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.core import Core, CoreCached
from ezvtb_rt.cache import DBCacherMP, RAMCacher
from ezvtb_rt.rife import RIFECoreLinked
from ezvtb_rt.tha import THACoreCachedRAM, THACoreCachedVRAM
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import numpy_to_image_file, img_file_to_numpy, generate_video, thaimg_to_cvimg
import json
import cv2

def CorePerf():
    cuda.start_profiler()
    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))


    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))
    cuda.stop_profiler()

def CoreTestShow():
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    pose_data = pose_data[800:1000][::2] # using 10fps pose data

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in range(len(poses)):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            if i != 0:
                for output in outputs:
                    new_vid.append(output[:,:,:3].copy())
        outputs = core.inference(np.array(poses[0]).reshape(1,45))
        for output in outputs:
            new_vid.append(output[:,:,:3].copy())
        return new_vid

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp16_x2.mp4', 20)

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp16_x3.mp4', 30)

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp16_x4.mp4', 40)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x2.mp4', 20)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x3.mp4', 30)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x4.mp4', 40)

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp16_x2.mp4', 20)

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp16_x3.mp4', 30)

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp16_x4.mp4', 40)
    
    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x2.mp4', 20)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x3.mp4', 30)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x4.mp4', 40)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x2_fp32.mp4', 20)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x3_fp32.mp4', 30)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x4_fp32.mp4', 40)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x2_fp32.mp4', 20)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x3_fp32.mp4', 30)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x4_fp32.mp4', 40)
    
def CoreCachedPerf():
    tha_core = THACoreCachedVRAM('./data/tha3/seperable/fp16', 2)
    rife_core = RIFECoreLinked('./data/rife_lite_v4_25/x3/fp16', tha_core)
    cacher = DBCacherMP()
    core = CoreCached(tha_core, cacher, rife_core)
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for pose in tqdm(pose_data):
        ret = core.inference(np.array(pose).reshape(1,45))
        for item in ret:
            item.copy()
    for pose in tqdm(pose_data):
        ret = core.inference(np.array(pose).reshape(1,45))
        for item in ret:
            item.copy()

def CoreCacheShow():
    tha_core = THACoreCachedVRAM('./data/tha3/seperable/fp16')
    rife_core = RIFECoreLinked('./data/rife_lite_v4_25/x2/fp16', tha_core)
    cacher = DBCacherMP()
    core = CoreCached(tha_core, cacher, rife_core)
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    pose_data = pose_data[800:1000]

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in range(len(poses)):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        for i in range(len(poses)):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        return new_vid
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/cached_core_half_sepe_fp16_x2_fp16.mp4', 40)
    print(core.cacher.hits, core.cacher.miss)

if __name__ == "__main__":
    os.makedirs('./test/data/core', exist_ok=True)
    CorePerf()
    CoreTestShow()
    CoreCacheShow()
    CoreCachedPerf()