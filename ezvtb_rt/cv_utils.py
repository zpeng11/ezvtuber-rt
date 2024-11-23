import sys
import os
sys.path.append(os.getcwd())
import cv2
import numpy as np
from typing import List
 
def numpy_srgb_to_linear(x:np.ndarray):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def numpy_linear_to_srgb(x:np.ndarray):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def cvimg_to_thaimg(im:np.ndarray, dtype:np.dtype = np.float32):
    shapes = im.shape
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(dtype)/255.0 #To RGB float
    im[:,:,:3]= numpy_srgb_to_linear(im[:,:,:3]) #To linear
    im = im * 2.0 - 1.0 # Normalize to 0
    im = im.reshape(shapes[0] * shapes[1], shapes[2]).transpose().reshape(1, shapes[2], shapes[0], shapes[1]) # Transpose to tensor shape
    return im

def thaimg_to_cvimg(im:np.ndarray, alpha:bool = False):
    shapes = im.shape
    im = im.reshape(shapes[1], shapes[2] * shapes[3]).transpose().reshape(shapes[2], shapes[3], shapes[1])
    im = (im + 1.0) / 2.0
    im[:,:,:3]= numpy_linear_to_srgb(im[:,:,:3])
    im = (im * 255.0).clip(0.0, 255.0).astype(np.uint8)
    if alpha:
        if shapes[1] == 4:
            return cv2.cvtColor(im, cv2.COLOR_RGBA2BGRA)
        else:
            return cv2.cvtColor(im, cv2.COLOR_RGB2BGRA)
    else:
        if shapes[1] == 4:
            return cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        else:
            return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def img_file_to_numpy(path:str, dtype:str = 'fp32'): # Image file to tha image
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 3: #RGB format
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    im = cvimg_to_thaimg(im)
    if 'fp16' in dtype:
        im = im.astype(np.float16)
    return im

def numpy_to_image_file(im:np.ndarray, path:str): # tha image to image file
    im = thaimg_to_cvimg(im, True if 'png' in path else False)
    cv2.imwrite(path, im)


# Function to generate video
def generate_video(imgs:List[np.ndarray], video_path:str, framerate:float): #Images should be prepared to be opencv image layout

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (512, 512))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")

    # Appending images to video
    for i in range(len(imgs)):
        video.write(imgs[i])

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
