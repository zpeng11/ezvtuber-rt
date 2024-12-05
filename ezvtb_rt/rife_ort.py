import sys
import os
sys.path.append(os.getcwd())
import onnxruntime as ort
import onnx
from typing import List
import numpy as np

class RIFEORTCore:
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

        self.rife = ort.InferenceSession(rife_dir+".onnx", sess_options=options, providers=providers, provider_options=provider_options)
        self.previous_frame = None

    def inference(self, img:np.ndarray):
        if self.previous_frame is None:
            self.previous_frame = img
        ret = self.rife.run(None, {
            'tha_img_0': self.previous_frame,
            'tha_img_1': img
        })
        self.previous_frame = img.copy()
        return ret

        