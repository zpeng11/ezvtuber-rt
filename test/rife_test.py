import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import check_build_all_models
from ezvtb_rt.rife import RIFECoreSimple, RIFECore
from ezvtb_rt.tha import THACoreSimple, THACore
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from tqdm import tqdm
from ezvtb_rt.cv_utils import generate_video
import json
import cv2


def RIFETestPerf():
    
    img1 = np.random.rand(1024,1024, 4).astype(np.uint8)
    img2 = np.random.rand(1024,1024, 4).astype(np.uint8)
    cuda.start_profiler()
    core = RIFECoreSimple('./data/rife_1024/x2/fp16')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_1024/x2/fp32')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_1024/x3/fp16')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_1024/x3/fp32')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_1024/x4/fp16')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_1024/x4/fp32')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    cuda.stop_profiler()

def RIFETestShow():
    core = THACoreSimple('./data/tha3/seperable/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45))
        tha_res.append(img.copy())
    tha_res = tha_res[1:]
    img = core.inference(np.array(pose).reshape(1,45))
    tha_res.append(img.copy())
    bgr_prepared = []
    for im in tha_res:
        bgr_prepared.append(im[:,:,:3])
    generate_video(bgr_prepared, './test/data/rife/base.mp4', 20)
    generate_video(bgr_prepared[::2], './test/data/rife/halfbase.mp4', 10)

    def createInterpolatedVideo(old_vid, core):
        new_vid = []
        for i in range(len(old_vid)):
            if i == 0:
                core.run(old_vid[0], old_vid[1])
                new_vid.append(old_vid[0].copy()[:,:,:3])
            elif i+1 <len(old_vid):
                interpolates = core.run(old_vid[i], old_vid[i+1])
                for inter in interpolates:
                    new_vid.append(inter.copy()[:,:,:3])
        interpolates = core.run(old_vid[0], old_vid[0])
        for inter in interpolates:
            new_vid.append(inter.copy()[:,:,:3])
        return new_vid

    core = RIFECoreSimple('./data/rife_512/x2/fp16')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x2.mp4', 40)

    core = RIFECoreSimple('./data/rife_512/x3/fp16')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x3.mp4', 60)

    core = RIFECoreSimple('./data/rife_512/x4/fp16')
    new_vid = createInterpolatedVideo(tha_res, core)
    generate_video(new_vid, './test/data/rife/x4.mp4', 80)

    core = RIFECoreSimple('./data/rife_512/x2/fp16')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx2.mp4', 20)

    core = RIFECoreSimple('./data/rife_512/x3/fp16')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx3.mp4', 30)

    core = RIFECoreSimple('./data/rife_512/x4/fp16')
    new_vid = createInterpolatedVideo(tha_res[::2], core)
    generate_video(new_vid, './test/data/rife/halfx4.mp4', 40)

if __name__ == "__main__":
    check_build_all_models()
    os.makedirs('./test/data/rife', exist_ok=True)
    RIFETestPerf()
    RIFETestShow()