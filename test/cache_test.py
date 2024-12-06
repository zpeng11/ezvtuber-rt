import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.cache import RAMCacher, DBCacherMP
import numpy  as np
import cv2
from tqdm import tqdm
import time
import random

def RAMCacherTest():
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    buf = np.ndarray((512, 512, 4), dtype=np.uint8)
    cacher = RAMCacher(1)
    for i in tqdm(range(10000)):
        cacher.write(i, img)
    for i in tqdm(range(10000)):
        buf[:,:,:3] = cacher.read(i)[:,:,:3]
    cv2.imwrite('./test/data/cache/ram_cacher.jpg', cacher.read(0))

def DBCacherTest():
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    buf = np.ndarray((512, 512, 4), dtype=np.uint8)
    cacher = DBCacherMP(cache_quality=90)
    cacher.write(0, img)
    for i in tqdm(range(1000)):
        time.sleep(0.02)
        cacher.read(random.randint(-1000000000, 1000000000))
        cacher.write(i, img)
    time.sleep(1)
    for i in tqdm(range(1000)):
        ret = cacher.read(i)
        if ret is not None:
            buf[:,:,:3] = ret[:,:,:3]
    if cacher.read(0) is not None:
        cv2.imwrite('./test/data/cache/db_cacher.jpg', cacher.read(0)[:,:,:3])

if __name__ == '__main__':
    os.makedirs('./test/data/cache', exist_ok=True)
    # RAMCacherTest()
    DBCacherTest()