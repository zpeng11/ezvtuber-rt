import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.tha_ort import THAORTCore, THAORTCoreNonDefault
import cv2
from tqdm import tqdm
import numpy as np
from ezvtb_rt.cv_utils import generate_video
import json

def THAORTTestPerf():
    core = THAORTCore('./data/tha3/seperable/fp16')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCore('./data/tha3/seperable/fp32')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCore('./data/tha3/standard/fp16')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCore('./data/tha3/standard/fp32')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCoreNonDefault('./data/tha3/seperable/fp16', 1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCoreNonDefault('./data/tha3/seperable/fp32', 1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCoreNonDefault('./data/tha3/standard/fp16', 1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

    core = THAORTCoreNonDefault('./data/tha3/standard/fp32', 1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    rand_pose = np.random.rand(1, 45).astype(np.float32)
    for i in tqdm(range(1000)):
        core.inference(rand_pose)

def THAORTTestShow():
    core = THAORTCore('./data/tha3/seperable/fp16')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/sepe16.mp4', 20)

    core = THAORTCore('./data/tha3/seperable/fp32')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/sepe32.mp4', 20)

    core = THAORTCore('./data/tha3/standard/fp16')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/stand16.mp4', 20)

    core = THAORTCore('./data/tha3/standard/fp32')
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/stand32.mp4', 20)


    core = THAORTCoreNonDefault('./data/tha3/seperable/fp16',0)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for i, pose in enumerate(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/sepe16_dev0.mp4', 20)

    core = THAORTCoreNonDefault('./data/tha3/seperable/fp16',1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/sepe16_dev1.mp4', 20)

    core = THAORTCoreNonDefault('./data/tha3/seperable/fp32',1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/sepe32_dev1.mp4', 20)

    core = THAORTCoreNonDefault('./data/tha3/standard/fp16',1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/stand16_dev1.mp4', 20)

    core = THAORTCoreNonDefault('./data/tha3/standard/fp32',1)
    core.update_image(cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    tha_res = []
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    core.update_image(cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED))
    for pose in tqdm(pose_data[800:1000]):
        img = core.inference(np.array(pose).reshape(1,45).astype(np.float32)).copy()
        tha_res.append(img[:,:,:3])
    generate_video(tha_res, './test/data/tha_ort/stand32_dev1.mp4', 20)

if __name__ == "__main__":
    os.makedirs('./test/data/tha_ort', exist_ok=True)
    THAORTTestPerf()
    THAORTTestShow()