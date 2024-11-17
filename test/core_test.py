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


    











# def THAWithRifePerf():
#     core = THAWithRIFE('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x4', 'rife_512')
#     core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
#     for i in tqdm(range(1000)):
#         core.inference(np.random.rand(1,45).astype(np.float16))


# if __name__ == "__main__":
