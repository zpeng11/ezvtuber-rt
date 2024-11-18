import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.core import Core
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import numpy_to_image_file, img_file_to_numpy, generate_video, thaimg_to_cvimg
import json

def CorePerf():
    cuda.start_profiler()
    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x2')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x3')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x4')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x2')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x3')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x4')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))


    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/rife_x2')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/rife_x3')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/rife_x4')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x2')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x3')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x4')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x2_fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x3_fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x4_fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x2_fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x3_fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45))

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x4_fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
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
                    new_vid.append(output[:,:,:3].astype(np.uint8))
        outputs = core.inference(np.array(poses[0]).reshape(1,45))
        for output in outputs:
            new_vid.append(output[:,:,:3].astype(np.uint8))
        return new_vid

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x2')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp16_x2.mp4', 20)

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x3')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp16_x3.mp4', 30)

    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x4')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp16_x4.mp4', 40)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x2')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x2.mp4', 20)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x3')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x3.mp4', 30)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x4')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x4.mp4', 40)

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/rife_x2')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp16_x2.mp4', 20)

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/rife_x3')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp16_x3.mp4', 30)

    core = Core('./data/tha3/standard/fp16', './data/rife_lite_v4_25/rife_x4')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp16_x4.mp4', 40)
    
    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x2')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x2.mp4', 20)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x3')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x3.mp4', 30)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x4')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x4.mp4', 40)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x2_fp32')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x2_fp32.mp4', 20)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x3_fp32')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x3_fp32.mp4', 30)

    core = Core('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x4_fp32')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_sepe_fp32_x4_fp32.mp4', 40)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x2_fp32')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x2_fp32.mp4', 20)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x3_fp32')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x3_fp32.mp4', 30)

    core = Core('./data/tha3/standard/fp32', './data/rife_lite_v4_25/rife_x4_fp32')
    core.setImage(img_file_to_numpy('./test/data/base.png'))
    new_vid = createInterpolatedVideo(pose_data, core)
    generate_video(new_vid, './test/data/core/half_stan_fp32_x4_fp32.mp4', 40)
    

if __name__ == "__main__":
    os.makedirs('./test/data/core', exist_ok=True)
    CorePerf()
    CoreTestShow()