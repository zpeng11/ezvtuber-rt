import onnx
import os

def check_exist_all_models():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, '..','data')
    rife_types = ['x2','x3','x4']
    rife_dtypes = ['fp32','fp16']
    rife_list = []
    for rife_type in rife_types:
        for rife_dtype in rife_dtypes:
            onnx_file = os.path.join(data_dir, 'rife_lite_v4_25',rife_type, rife_dtype+'.onnx')
            if not os.path.isfile(onnx_file):
                raise ValueError('Data is not prepared')
            onnx.checker.check_model(onnx_file)
            rife_list.append(onnx_file)
    tha_types = ['seperable', 'standard']
    tha_dtypes = ['fp32', 'fp16']
    tha_components = ['combiner.onnx', 'decomposer.onnx','editor.onnx', 'morpher.onnx', 'rotator.onnx']
    tha_list = []
    for tha_type in tha_types:
        for tha_dtype in tha_dtypes:
            for tha_component in tha_components:
                onnx_file = os.path.join(data_dir, 'tha3', tha_type, tha_dtype, tha_component)
                if not os.path.isfile(onnx_file):
                    raise ValueError('Data is not prepared')
                onnx.checker.check_model(onnx_file)
                tha_list.append(onnx_file)
    return rife_list + tha_list