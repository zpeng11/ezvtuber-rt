import onnxruntime as ort
import numpy as np
from typing import List

class SRORTCore:
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

        self.sr = ort.InferenceSession(model_dir+".onnx", sess_options=options, providers=providers, provider_options=provider_options)

        self.input_name = 'x' if 'waifu2x' in model_dir else 'input'
            

    def inference(self, imgs:List[np.ndarray])-> List[np.ndarray]:
        ret = []
        for img in imgs:
            ret.append(self.sr.run(None, {self.input_name: img})[0])
        return ret