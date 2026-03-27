# Installation Guide

This guide explains how to install `ezvtuber-rt` as a Python library. You need to install Anaconda beforehead and prepare a Python3.10 Environment

## Download TensorRT-RTX
1. Go to https://developer.nvidia.com/tensorrt-rtx download for windows cuda129
2. Unzip downloaded folder
3. Add bin folder to environment PATH
4. Activate your python environment and install like below:
```bash
pip install D:\TensorRT-RTX-1.4.0.76_cu129\python\tensorrt_rtx-1.4.0.76-cp310-none-win_amd64.whl
```

## Installation Methods
```bash
git clone https://github.com/zpeng11/ezvtuber-rt.git && cd ezvtuber-rt
conda install conda-forge::pycuda
conda install -c nvidia/label/cuda-12.9.1 cuda-nvcc-dev_win-64 cudnn cuda-runtime
pip install . 
```

You can also do development install, requirements are the same
```bash
pip install -e .
```



## Basic Usage
### Download Model Data
Download model data from [release]() and extract it to `data` folder for default, or you will need to explicitly provide the path when using
### Environment Variables
* `set EZVTB_DEVICE_ID=0`(other than 0 for other GPU) optional to indicate if want to run on non-default GPU

### Using ONNX Runtime DirectML (ALL GPUs)

```python
from ezvtb_rt import CoreORT
from ezvtb_rt import init_model_path

# Initialize the model path
init_model_path('C:/Path/To/Model/Folder')

# Initialize the core with TensorRT backend
core = CoreTRT()

#setup input image
core.setImage(cv2_bgra_image)
# Use the core for inference
while True:
    pose: np.ndarray = np.array(get_pose()).astype(np.float32).reshape((1,45)) #Get pose from sensor
    results: np.ndarray = core.inference([pose])
    #This results is a (N, H, W, 4) BGRA image, display or port to streaming

```

### Using TensorRT (NVIDIA GPUs)


```python
from ezvtb_rt import CoreTRT
from ezvtb_rt import init_model_path
from ezvtb_rt.trt_utils import check_build_all_models()

# Initialize the model path
init_model_path('C:/Path/To/Model/Folder')

# Initialize the core with TensorRT backend
core = CoreTRT()

#setup input image
core.setImage(cv2_bgra_image)
# Use the core for inference
while True:
    pose: np.ndarray = np.array(get_pose()).astype(np.float32).reshape((1,45)) #Get pose from sensor
    results: np.ndarray = core.inference(pose)
    #This results is a (N, H, W, 4) BGRA image, display or port to streaming
```



## Feature-focused examples

The snippets below mirror the cases covered in the automated tests (see `test/core_test.py`) and show how to enable specific features one at a time or in combination.

### THA-only (default path)
```python
import cv2
import numpy as np
from ezvtb_rt import CoreTRT, init_model_path

init_model_path('C:/Path/To/Model/Folder')
core = CoreTRT(tha_model_version='v3', tha_model_seperable=True, tha_model_fp16=True, cache_max_giga=2.0)

img = cv2.imread('path/to/base.png', cv2.IMREAD_UNCHANGED)
core.setImage(img)

pose = np.array(get_pose()).astype(np.float32).reshape((1, 45))
frame:np.ndarray = core.inference([pose])  # (1, 512, 512, 4)
```

### THA v4
```python
core = CoreTRT(tha_model_version='v4', tha_model_fp16=True, cache_max_giga=0.0)
core.setImage(cv2.imread('path/to/base.png', cv2.IMREAD_UNCHANGED))
pose = np.array(get_pose()).astype(np.float32).reshape((1, 45))
frame = core.inference([pose])  # same shape as v3, higher-quality models
```

### RIFE interpolation
```python
core = CoreTRT(
    tha_model_version='v3',
    tha_model_seperable=True,
    tha_model_fp16=True,
    rife_model_enable=True,
    rife_model_scale=3,
    rife_model_fp16=True,
    cache_max_giga=2.0,
)
core.setImage(cv2.imread('path/to/base.png', cv2.IMREAD_UNCHANGED))

prev_pose = None
while True:  # processing loop
    curr_pose = np.array(get_pose()).astype(np.float32).reshape((1, 45))
    if prev_pose is None:
        prev_pose = curr_pose
    pose_0_33 = prev_pose + (prev_pose + curr_pose)/3  # Pose interpolation on 1/3
    pose_0_66 = prev_pose + 2*(prev_pose + curr_pose)/3 # Pose interpolation on 2/3
    #You may need to qunatize poses for simplification in order to hit catch of poses.
    frames: np.ndarray = core.inference([pose_0_33, pose_0_66, curr_pose])  # shape: (3, 512, 512, 4)
    prev_pose = curr_pose
    #Timing control here
```

### Super resolution (SR)
```python
core = CoreTRT(
    tha_model_version='v3',
    tha_model_seperable=True,
    tha_model_fp16=True,
    sr_model_enable=True,
    sr_model_scale=2,  # use 4 for Real-ESRGAN x4
    sr_model_fp16=True,
    cache_max_giga=2.0,
)
core.setImage(cv2.imread('path/to/base.png', cv2.IMREAD_UNCHANGED))

pose = np.array(get_pose()).astype(np.float32).reshape((1, 45))
sr_frame: np.ndarray = core.inference([pose])  # shape: (1, 1024, 1024, 4) for x2
```

### Adjust Caching (reuse same pose)
```python
core = CoreTRT(
    tha_model_version='v3',
    tha_model_seperable=True,
    tha_model_fp16=True,
    vram_cache_size = 1.0, #Enable vram cache, this actually stores in gpu mapped RAM instead of vram
    cache_max_giga=1.0,  # enable cpu cache
)
core.setImage(cv2.imread('path/to/base.png', cv2.IMREAD_UNCHANGED))

pose = np.array(get_pose()).astype(np.float32).reshape((1, 45))
# It is necessary for the user of the library to quantize poses to hit caches
first: np.ndarray = core.inference([pose])
second: np.ndarray = core.inference([pose])
print(f"hits={core.cacher.hits}, miss={core.cacher.miss}")
```

### Combined RIFE + SR
```python
core = CoreTRT(
    tha_model_version='v3',
    tha_model_seperable=True,
    tha_model_fp16=True,
    rife_model_enable=True,
    rife_model_scale=2,
    rife_model_fp16=True,
    sr_model_enable=True,
    sr_model_scale=4,
    sr_model_fp16=True,
    cache_max_giga=2.0,
)
core.setImage(cv2.imread('path/to/base.png', cv2.IMREAD_UNCHANGED))

prev_pose = None
while True:  # processing loop
    curr_pose = np.array(get_pose()).astype(np.float32).reshape((1, 45))
    if prev_pose is None:
        prev_pose = curr_pose
    pose_0_5 = prev_pose + (prev_pose + curr_pose)/2
    #You may need to qunatize poses for simplification in order to hit catch of poses.
    frames: np.ndarray = core.inference([prev_pose, curr_pose])  # shape: (2, 1024, 1024, 4)
    prev_pose = curr_pose
```