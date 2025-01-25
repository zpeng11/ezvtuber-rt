import onnxruntime as ort
import numpy as np
from typing import List

class SRORT:
    def __init__(self, model_dir:str, device_id:int):
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

        self.sr = ort.InferenceSession(model_dir, sess_options=options, providers=providers, provider_options=provider_options)

        self.input_name = 'x' if 'waifu2x' in model_dir else 'input'
            

    def inference(self, imgs:List[np.ndarray])-> List[np.ndarray]:
        ret = []
        for img in imgs:
            ret.append(self.sr.run(None, {self.input_name: img})[0])
        return ret

if __name__ == '__main__':
    import cv2
    import os
    
    # Initialize SR with waifu2x model
    sr = SRORT('data/models/Real-ESRGAN/exported_256_fp16.onnx', 0)
    
    # Load test image
    img = cv2.imread('./data/images/lambda_00.png', cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError('Test image not found in data/images directory')
    
    # Show original image
    cv2.imshow('Original', img)
    cv2.waitKey(500)
    
    print(f'Running super-resolution ')
    result = sr.inference([img])[0]
    cv2.imshow(f'Super-Resolved ', result)
    cv2.waitKey(500)

    # Initialize SR with waifu2x model
    sr = SRORT('data\\models\\waifu2x_upconv\\fp16\\upconv_7\\photo\\noise1_scale2x.onnx', 0)
    
    # Load test image
    img = cv2.imread('./data/images/lambda_00.png', cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError('Test image not found in data/images directory')
    
    # Show original image
    cv2.imshow('Original', img)
    cv2.waitKey(1000)
    
    print(f'Running super-resolution ')
    result = sr.inference([img])[0]
    cv2.imshow(f'Super-Resolved ', result)
    cv2.waitKey(1000)
    
    cv2.destroyAllWindows()
