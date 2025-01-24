import onnxruntime as ort
import numpy as np
from typing import List

class AffineORT:
    def __init__(self, model_dir:str, device_id:int, angle:int, translate:np.ndarray, scale:int):
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

        self.sess = ort.InferenceSession(model_dir, sess_options=options, providers=providers, provider_options=provider_options)

        self.input_name = 'x' if 'waifu2x' in model_dir else 'input'
        self.angle = np.array([angle,],dtype=np.float32)
        self.translate = translate
        self.scale = np.array([scale,],dtype=np.float32)
            

    def inference(self, img:List[np.ndarray])-> List[np.ndarray]:
        return self.sess.run(None,{
            'img':img[0],
            'angle':self.angle,
            'translate':self.translate,
            'scale':self.scale
        })
    
if __name__ == '__main__':
    import cv2
    affine = AffineORT('Z:\\talking-head-anime-3-demo\\affine\\affine_512.onnx', 0, 11, [-20,20], 1.1)
    image = cv2.imread('./data/images/lambda_00.png', cv2.IMREAD_UNCHANGED)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image', affine.inference(image)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    