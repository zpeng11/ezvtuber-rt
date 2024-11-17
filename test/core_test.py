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
    core = Core('./data/tha3/seperable/fp16', './data/rife_lite_v4_25/rife_x4')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    cuda.start_profiler()
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45).astype(np.float16))
    cuda.stop_profiler()


if __name__ == "__main__":
    CorePerf()