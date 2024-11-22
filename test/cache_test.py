import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.cache import JPEG2THARes, THARes2JPEG
from ezvtb_rt.cv_utils import img_file_to_numpy, numpy_to_image_file
import numpy  as np


def main():
    tha_img = img_file_to_numpy('./test/data/base.png', 'fp16')
    encoded = THARes2JPEG(tha_img[:,:3,:,:], 90)
    decoded = JPEG2THARes(encoded)
    tha_img[:,:3,:,:] = decoded
    numpy_to_image_file(tha_img, './test/data/cache/jpeg_compressed.png')

if __name__ == '__main__':
    os.makedirs('./test/data/cache', exist_ok=True)
    main()