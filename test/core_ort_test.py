import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.init_utils import check_exist_all_models
from ezvtb_rt.core_ort import CoreORT
from ezvtb_rt.tha_ort import THAORTCore
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import generate_video
from ezvtb_rt.cache import Cacher
import json
import cv2

  
def CoreORTPerf():
    core = CoreORT('./data/tha3/seperable/fp32', rife_path='./data/rife_512/x4/fp32', device_id=1)#, cacher=Cacher(image_size=512), sr_path='data\\Real-ESRGAN\\exported_256_fp16', rife_path='./data/rife_512/x2/fp32',  device_id=1, cacher=Cacher(image_size=1024))
    core.setImage( cv2.imread('f:/talking-head-anime-3-demo/data/images/crypko_01.png', cv2.IMREAD_UNCHANGED))
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

def CoreORTTestShow():
    core = CoreORT('./data/tha3/seperable/fp16',sr_path='data\\Real-ESRGAN\\exported_256_fp16', device_id=1, cacher=Cacher(image_size=512))#, sr_path='data\\Real-ESRGAN\\exported_256_fp16', rife_path='./data/rife_512/x2/fp32',  device_id=1, cacher=Cacher(image_size=1024))
    core.setImage( cv2.imread('f:/talking-head-anime-3-demo/data/images/crypko_01.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3])
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3])
        return new_vid
    
    vid = createInterpolatedVideo(pose_data[800:1000], core)
    generate_video(vid, './test/data/core_ort/test.mp4', 20)
    if core.cacher is not None:
        print(core.cacher.hits, core.cacher.miss)



if __name__ == "__main__":
    # check_exist_all_models()
    # CoreORTPerf()
    CoreORTTestShow()