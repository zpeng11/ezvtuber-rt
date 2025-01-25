import onnxruntime as ort
import numpy as np
from typing import List

class RIFEORT:
    #We can not avoid copying in cpu with ort python api, so no need to use io binding
    def __init__(self, rife_dir:str, device_id:int = 0):
        if 'x2' in rife_dir:
            self.scale = 2
        elif 'x3' in rife_dir:
            self.scale = 3
        elif 'x4' in rife_dir:
            self.scale = 4
        else:
            raise ValueError('can not determine scale')

        avaliales = ort.get_available_providers()
        if 'CUDAExecutionProvider' in avaliales:
            self.provider = 'CUDAExecutionProvider'
            self.device = 'cuda'
        elif 'DmlExecutionProvider' in avaliales:
            self.provider = 'DmlExecutionProvider'
            self.device = 'dml'
        else:
            raise ValueError('Please check environment, ort does not have available gpu provider')
        print('Using EP:', self.provider)

        providers = [ self.provider]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider_options = [{'device_id':device_id}]

        self.rife = ort.InferenceSession(rife_dir + '.onnx', sess_options=options, providers=providers, provider_options=provider_options)
        self.previous_frame = None

    def inference(self, img:List[np.ndarray]) -> List[np.ndarray]:
        if self.previous_frame is None:
            self.previous_frame = img[0]
        ret = self.rife.run(None, {
            'tha_img_0': self.previous_frame,
            'tha_img_1': img[0]
        })
        self.previous_frame = ret[-1]
        return ret

if __name__ == '__main__':
    import cv2
    import os
    
    # Initialize RIFE with x2 model
    rife = RIFEORT('C:\\EasyVtuber\\data\\models\\rife_512\\x2\\fp16', 0)
    
    # Load two test images
    img1 = cv2.imread('./data/images/lambda_00.png', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('./data/images/lambda_01.png', cv2.IMREAD_UNCHANGED)
    imgs = [img1, img2]
    
    if img1 is None or img2 is None:
        raise FileNotFoundError('Test images not found in data/images directory')
    
    # Show original frames
    cv2.imshow('Frame 1', img1)
    cv2.waitKey(500)
    cv2.imshow('Frame 2', img2)
    cv2.waitKey(500)
    
    # Run multiple interpolation iterations
    for i in range(10):  # Number of interpolation iterations
        print(f'Running interpolation iteration {i+1}')
        result = rife.inference([imgs[i%2]])
        cv2.imshow(f'Interpolated Frame {i+1}', result[0])
        cv2.waitKey(500)
        cv2.imshow(f'Interpolated Frame {i+1}', result[1])
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()
