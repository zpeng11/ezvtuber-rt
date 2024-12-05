import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.rife_ort import RIFEORTCore
from ezvtb_rt.tha_ort import THAORTCore
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import numpy_to_image_file, img_file_to_numpy, generate_video, thaimg_to_cvimg
import json
import cv2


def RIFEORTTestPerf():
    
    img1 = np.random.rand(512,512, 4).astype(np.uint8)
    cuda.start_profiler()
    core = RIFEORTCore('./data/rife_lite_v4_25/x2/fp16',0)
    for i in tqdm(range(1000)):
        ret = core.inference(img1)

    core = RIFEORTCore('./data/rife_lite_v4_25/x2/fp32',0)
    for i in tqdm(range(1000)):
        ret = core.inference(img1)

    core = RIFEORTCore('./data/rife_lite_v4_25/x3/fp16',0)
    for i in tqdm(range(1000)):
        ret = core.inference(img1)

    core = RIFEORTCore('./data/rife_lite_v4_25/x3/fp32',0)
    for i in tqdm(range(1000)):
        ret = core.inference(img1)

    core = RIFEORTCore('./data/rife_lite_v4_25/x4/fp16',0)
    for i in tqdm(range(1000)):
        ret = core.inference(img1)

    core = RIFEORTCore('./data/rife_lite_v4_25/x4/fp32',0)
    for i in tqdm(range(1000)):
        ret = core.inference(img1)

    cuda.stop_profiler()

def RIFEORTTestShow():
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    core = THAORTCore('./data/tha3/seperable/fp16')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img)
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img)
    # generate_video(tha_res, './test/data/rife_ort/base.mp4', 20)
    bgr_prepared = []
    for im in tha_res:
        bgr_prepared.append(im[:,:,:3])
    generate_video(bgr_prepared, './test/data/rife_ort/base.mp4', 20)
    generate_video(bgr_prepared[::2], './test/data/rife_ort/halfbase.mp4', 10)

    def createInterpolatedVideo(old_vid, core):
        new_vid = []
        for i in range(len(old_vid)):
            interpolates = core.inference(old_vid[i])
            for inter in interpolates:
                new_vid.append(inter.copy()[:,:,:3])
        return new_vid

    core = RIFEORTCore('./data/rife_lite_v4_25/x2/fp16')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife_ort/x2.mp4', 40)

    core = RIFEORTCore('./data/rife_lite_v4_25/x3/fp16')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife_ort/x3.mp4', 60)

    core = RIFEORTCore('./data/rife_lite_v4_25/x4/fp16')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife_ort/x4.mp4', 80)

    core = RIFEORTCore('./data/rife_lite_v4_25/x2/fp16')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife_ort/halfx2.mp4', 20)

    core = RIFEORTCore('./data/rife_lite_v4_25/x3/fp16')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife_ort/halfx3.mp4', 30)

    core = RIFEORTCore('./data/rife_lite_v4_25/x4/fp16')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife_ort/halfx4.mp4', 40)

if __name__ == "__main__":
    os.makedirs('./test/data/rife_ort', exist_ok=True)
    RIFEORTTestPerf()
    RIFEORTTestShow()