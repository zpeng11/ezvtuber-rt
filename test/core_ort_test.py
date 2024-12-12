import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.init_utils import check_exist_all_models
from ezvtb_rt.core_ort import CoreORT, CoreORTCached
from ezvtb_rt.tha_ort import THAORTCore
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import generate_video
from ezvtb_rt.cache import RAMCacher, DBCacherMP
import json
import cv2

def CoreORTPerf():
    core = CoreORT('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))


    core = CoreORT('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage(np.random.rand(512,512, 4).astype(np.uint8))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

def CoreORTTestShow():
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    pose_data = pose_data[800:1000][::2] # using 10fps pose data

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in range(len(poses)):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        return new_vid

    core = CoreORT('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp16_x2.mp4', 20)

    core = CoreORT('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp16_x3.mp4', 30)

    core = CoreORT('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp16_x4.mp4', 40)

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp32_x2.mp4', 20)

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp32_x3.mp4', 30)

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp32_x4.mp4', 40)

    core = CoreORT('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp16_x2.mp4', 20)

    core = CoreORT('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp16_x3.mp4', 30)

    core = CoreORT('./data/tha3/standard/fp16', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp16_x4.mp4', 40)
    
    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp32_x2.mp4', 20)

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp32_x3.mp4', 30)

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp32_x4.mp4', 40)

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp32_x2_fp32.mp4', 20)

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp32_x3_fp32.mp4', 30)

    core = CoreORT('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_sepe_fp32_x4_fp32.mp4', 40)

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x2/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp32_x2_fp32.mp4', 20)

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x3/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp32_x3_fp32.mp4', 30)

    core = CoreORT('./data/tha3/standard/fp32', './data/rife_lite_v4_25/x4/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core_ort/half_stan_fp32_x4_fp32.mp4', 40)
    
def CoreORTCachedPerf():
    cacher = DBCacherMP()
    core = CoreORTCached('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x3/fp32', cacher=cacher)
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

def CoreORTCachedShow():
    core = CoreORTCached('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/x2/fp32', cacher=RAMCacher(cache_quality=99))
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        return new_vid
    
    vid = createInterpolatedVideo(pose_data[800:1000], core)
    generate_video(vid, './test/data/core_ort/cached_sepe_fp32_x3_fp32.mp4', 40)
    print(core.cacher.hits)

def cacher_debug():
    cacher = DBCacherMP(cache_quality=99)
    core = CoreORTCached('./data/tha3/seperable/fp32',rife_path='./data/rife_lite_v4_25/x3/fp32',cacher=cacher)
    # core = THAORTCore('./data/tha3/seperable/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    pose_data = pose_data[800:1000]
    vid1 = []
    for i in range(len(pose_data)):
        outputs = core.inference(np.array(pose_data[i]).reshape(1,45))
        for output in outputs:
            vid1.append(output[:,:,:3].copy())
    vid1 = vid1[2:]
    # core = THAORTCore('./data/tha3/seperable/fp32')
    # core.update_image( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    vid2 = []
    for i in range(len(pose_data)):
        outputs = core.inference(np.array(pose_data[i]).reshape(1,45))
        for output in outputs:
            vid2.append(output[:,:,:3].copy())
    vid2 = vid2[2:]

    error_sum = 0.0
    for i in range(len(vid1)):
        error_sum += np.abs(vid1[i] - vid2[i]).sum()/vid1[i].size
    print(error_sum/len(vid1))
    print(cacher.hits)

if __name__ == "__main__":
    check_exist_all_models()
    os.makedirs('./test/data/core_ort', exist_ok=True)
    db_path= './cacher.sqlite'
    if os.path.exists(db_path):
        os.remove(db_path)
    CoreORTPerf()
    CoreORTTestShow()
    CoreORTCachedPerf()
    CoreORTCachedShow()
    cacher_debug()