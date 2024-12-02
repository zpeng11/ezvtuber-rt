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
import cv2

def THATestPerf():
    core = THACoreSimple('./data/tha3/seperable/fp16')
    cuda.start_profiler()
    core.setImage(np.random.rand(512,512,4).astype(np.uint8))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float32))

    core = THACoreSimple('./data/tha3/seperable/fp32')
    core.setImage(np.random.rand(512,512,4).astype(np.uint8))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float32))

    core = THACoreSimple('./data/tha3/standard/fp16')
    core.setImage(np.random.rand(512,512,4).astype(np.uint8))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float32))

    core = THACoreSimple('./data/tha3/standard/fp32')
    core.setImage(np.random.rand(512,512,4).astype(np.uint8))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float32))
    cuda.stop_profiler()

def THATestShow(): #include test for face param in the future
    core = THACoreSimple('./data/tha3/seperable/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45))
        tha_res.append(img.copy()[:,:,:3])
    tha_res = tha_res[1:]
    img = core.inference(np.array(pose).reshape(1,45))
    tha_res.append(img.copy()[:,:,:3])
    generate_video(tha_res, './test/data/tha/sepe16.mp4', 20)

    core = THACoreSimple('./data/tha3/seperable/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45))
        tha_res.append(img.copy()[:,:,:3])
    tha_res = tha_res[1:]
    img = core.inference(np.array(pose).reshape(1,45))
    tha_res.append(img.copy()[:,:,:3])
    generate_video(tha_res, './test/data/tha/sepe32.mp4', 20)
    

    core = THACoreSimple('./data/tha3/standard/fp32')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45))
        tha_res.append(img.copy()[:,:,:3])
    tha_res = tha_res[1:]
    img = core.inference(np.array(pose).reshape(1,45))
    tha_res.append(img.copy()[:,:,:3])
    generate_video(tha_res, './test/data/tha/stand32.mp4', 20)

    core = THACoreSimple('./data/tha3/standard/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45))
        tha_res.append(img.copy()[:,:,:3])
    tha_res = tha_res[1:]
    img = core.inference(np.array(pose).reshape(1,45))
    tha_res.append(img.copy()[:,:,:3])
    generate_video(tha_res, './test/data/tha/stand16.mp4', 20)


if __name__ == "__main__":
    os.makedirs('./test/data/tha', exist_ok=True)
    THATestShow()
    THATestPerf()