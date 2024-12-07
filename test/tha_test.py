import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.rife import RIFECoreSimple, RIFECore
from ezvtb_rt.tha import THACoreSimple, THACoreCachedRAM, THACoreCachedVRAM
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
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for pose in tqdm(pose_data[:3000]):
        ret = core.inference(np.array(pose).reshape(1,45)).copy()

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

from ezvtb_rt.trt_utils import HostDeviceMem
def PCIECopyPerf():
    stream = cuda.Stream()
    def createMem(shape, dtype):
        host_mem = cuda.pagelocked_empty(shape, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return HostDeviceMem(host_mem, device_mem)
    full_fp = createMem((1,4,512,512), np.float32)
    for i in tqdm(range(100000)):
        full_fp.dtoh(stream)
        stream.synchronize()
    for i in tqdm(range(100000)):
        full_fp.htod(stream)
        stream.synchronize()  

def THACacheVRAMPerf():
    core = THACoreCachedVRAM('./data/tha3/seperable/fp16', 1)
    cuda.start_profiler()
    core.setImage(np.random.rand(512,512,4).astype(np.uint8))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for pose in tqdm(pose_data):
        ret = core.inference(np.array(pose).reshape(1,45),True).copy()
    print(core.combiner_cacher.hits, core.combiner_cacher.miss, core.morpher_cacher.hits, core.morpher_cacher.miss)
    cuda.stop_profiler()

def THACacheRAMPerf():
    core = THACoreCachedRAM('./data/tha3/seperable/fp16', 2)
    cuda.start_profiler()
    core.setImage(np.random.rand(512,512,4).astype(np.uint8))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for pose in tqdm(pose_data):
        ret = core.inference(np.array(pose).reshape(1,45),True).copy()
    print(core.hits, core.miss)
    cuda.stop_profiler()

def THACacheVRAMShow():
    core = THACoreCachedVRAM('./data/tha3/seperable/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45), True)
        tha_res.append(img.copy()[:,:,:3])

    tha_res_cached = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45), True)
        tha_res_cached.append(img.copy()[:,:,:3])

    mae = 0
    for i in range(len(tha_res)):
        mae += np.abs((tha_res[i] - tha_res_cached[i])).sum()
    print('cache mae:',mae)
    generate_video(tha_res_cached, './test/data/tha/sepe16_vram.mp4', 20)
    print(core.combiner_cacher.hits, core.combiner_cacher.miss, core.morpher_cacher.hits, core.morpher_cacher.miss)

def THACacheRAMShow():
    core = THACoreCachedRAM('./data/tha3/seperable/fp16')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45), True)
        tha_res.append(img.copy()[:,:,:3])

    tha_res_cached = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45), True)
        tha_res_cached.append(img.copy()[:,:,:3])

    mae = 0
    for i in range(len(tha_res)):
        mae += np.abs((tha_res[i] - tha_res_cached[i])).sum()
    print('cache mae:',mae)
    generate_video(tha_res_cached, './test/data/tha/sepe16_ram.mp4', 20)
    print(core.hits, core.miss)

if __name__ == "__main__":
    os.makedirs('./test/data/tha', exist_ok=True)
    # PCIECopyPerf()
    # THATestShow()
    # THATestPerf()
    # THACacheVRAMPerf()
    # THACacheVRAMShow()
    THACacheRAMPerf()
    # THACacheRAMShow()
    