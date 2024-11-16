import sys
import os
sys.path.append(os.getcwd())
import cv2
import numpy as np

def numpy_srgb_to_linear(x:np.ndarray):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def numpy_linear_to_srgb(x:np.ndarray):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def img_file_to_numpy(path:str, dtype:str = 'fp32'):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    channels = im.shape[2]
    assert(im.shape[0] == 512 and im.shape[1] == 512)
    if channels == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(np.float32)/255.0 #To RGB float
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 #To RGB float
    im[:,:,:3]= numpy_srgb_to_linear(im[:,:,:3]) #To linear
    im = im * 2.0 - 1.0 # Normalize to 0
    
    im = im.reshape(512 * 512, channels).transpose().reshape(1, channels, 512, 512) # Transpose to tensor shape
    if 'fp16' in dtype:
        im = im.astype(np.float16)
    return im

def numpy_to_image_file(im:np.ndarray, path:str):
    channels = im.shape[1]
    im = im.reshape(channels, 512 * 512).transpose().reshape(512, 512, channels)
    im = (im + 1.0) / 2.0
    im[:,:,:3]= numpy_linear_to_srgb(im[:,:,:3])
    im = (im * 255.0).clip(0.0, 255.0).astype(np.uint8)
    if channels == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, im)
