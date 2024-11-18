import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.rife import RIFECoreSimple, RIFECore
from ezvtb_rt.tha import THACoreSimple, THACore
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import numpy_to_image_file, img_file_to_numpy, generate_video, thaimg_to_cvimg
import json


def RIFETestPerf():
    
    img1 = np.random.rand(1,4,512,512).astype(np.float16)
    img2 = np.random.rand(1,4,512,512).astype(np.float16)
    cuda.start_profiler()
    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2_fp32')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3_fp32')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4_fp32')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    cuda.stop_profiler()

def RIFETestShow():
    core = THACoreSimple('./data/tha3/standard/fp32')
    core.setImage( img_file_to_numpy('./test/data/base.png'))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    base_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45))
        tha_res.append(img.copy())
        base_res.append(thaimg_to_cvimg(img.copy()))
    tha_res = tha_res[1:]
    base_res = base_res[1:]
    img = core.inference(np.array(pose).reshape(1,45))
    tha_res.append(img.copy())
    base_res.append(thaimg_to_cvimg(img.copy()))
    generate_video(base_res, './test/data/rife/base.mp4', 20)
    generate_video(base_res[::2], './test/data/rife/halfbase.mp4', 10)

    def createInterpolatedVideo(old_vid, core):
        new_vid = []
        for i in range(len(old_vid)):
            if i == 0:
                core.run(old_vid[0], old_vid[1])
                new_vid.append(thaimg_to_cvimg(old_vid[0])[:,:,:3])
            elif i+1 <len(old_vid):
                interpolates = core.run(old_vid[i], old_vid[i+1])
                for inter in interpolates:
                    new_vid.append(inter.astype(np.uint8)[:,:,:3])
        interpolates = core.run(old_vid[0], old_vid[0])
        for inter in interpolates:
            new_vid.append(inter.astype(np.uint8)[:,:,:3])
        return new_vid

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x2.mp4', 40)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x3.mp4', 60)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x4.mp4', 80)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx2.mp4', 20)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx3.mp4', 30)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx4.mp4', 40)


    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2_fp32')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x2_fp32.mp4', 40)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3_fp32')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x3_fp32.mp4', 60)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4_fp32')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x4_fp32.mp4', 80)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2_fp32')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx2_fp32.mp4', 20)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3_fp32')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx3_fp32.mp4', 30)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4_fp32')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx4_fp32.mp4', 40)

if __name__ == "__main__":
    os.makedirs('./test/data/rife', exist_ok=True)
    RIFETestPerf()
    RIFETestShow()