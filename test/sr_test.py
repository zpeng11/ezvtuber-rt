import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.sr import SR
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from tqdm import tqdm
import cv2
from ezvtb_rt.trt_utils import check_build_all_models
import onnxruntime as ort

def SRPerfTest():
    inp = np.zeros((512,512,4), dtype= np.uint8)
    core = SR('data\\Real-ESRGAN\\exported_256_fp16')
    for i in tqdm(range(10000)):
        core.run(inp)

    inp = np.zeros((512,512,4), dtype= np.uint8)
    core = SR('data\\Real-ESRGAN\\exported_256')
    for i in tqdm(range(10000)):
        core.run(inp)

    inp = np.zeros((512,512,4), dtype= np.uint8)
    core = SR('data\\waifu2x_upconv\\fp16\\upconv_7\\art\\noise0_scale2x')
    for i in tqdm(range(10000)):
        core.run(inp)

    inp = np.zeros((512,512,4), dtype= np.uint8)
    core = SR('data\\waifu2x_upconv\\fp32\\upconv_7\\art\\noise0_scale2x')
    for i in tqdm(range(10000)):
        core.run(inp)

def SRTestShow():
    img = cv2.imread("test/data/base_1.png", cv2.IMREAD_UNCHANGED)
    core = SR('data\\Real-ESRGAN\\exported_256_fp16')
    cv2.imwrite("test/data/sr/x4fp16.png", core.run(img))

    core = SR('data\\Real-ESRGAN\\exported_256')
    cv2.imwrite("test/data/sr/x4fp32.png", core.run(img))

    core = SR('data\\waifu2x_upconv\\fp16\\upconv_7\\art\\noise0_scale2x')
    cv2.imwrite("test/data/sr/x2fp16.png", core.run(img))

    core = SR('data\\waifu2x_upconv\\fp32\\upconv_7\\art\\noise0_scale2x')
    cv2.imwrite("test/data/sr/x2fp32.png", core.run(img))




if __name__ == "__main__":
    # check_build_all_models()
    os.makedirs('./test/data/sr', exist_ok=True)
    # SRPerfTest()
    SRTestShow()