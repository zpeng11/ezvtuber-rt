import onnx
import os
import urllib.request
import shutil

def prepare_download_models():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, '..','data')
    print('Downloading pretrained models...')
    zip_path = 'https://github.com/zpeng11/ezvtuber-rt/releases/download/0.0.1/20241220.zip'
    filehandle, _ = urllib.request.urlretrieve(zip_path)
    os.rename(filehandle, filehandle + '.zip')
    shutil.unpack_archive(filehandle + '.zip' , data_dir)

def check_exist_all_models():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, '..','data')

    if not os.path.exists(os.path.join(data_dir, 'tha3')):
        prepare_download_models()

    rife_types = ['x2','x3','x4']
    rife_dtypes = ['fp32','fp16']
    rife_list = []
    for rife_type in rife_types:
        for rife_dtype in rife_dtypes:
            onnx_file = os.path.join(data_dir, 'rife_512',rife_type, rife_dtype+'.onnx')
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
    real_esrgan_list = [os.path.join(data_dir,'Real-ESRGAN','exported_256_fp16.onnx'), os.path.join(data_dir,'Real-ESRGAN','exported_256.onnx')]
    waifu2x_dtypes = ['fp32','fp16']
    waifu2x_train_sources = ['art']
    waifu2x_models = ['noise0_scale2x.onnx', 
                      'noise1_scale2x.onnx', 
                      'noise2_scale2x.onnx',
                      'noise3_scale2x.onnx',
                      'scale2x.onnx']
    waifu2x_list = []
    for waifu2x_dtype in waifu2x_dtypes:
        for waifu2x_train_source in waifu2x_train_sources:
            for waifu2x_model in waifu2x_models:
                onnx_file = os.path.join(data_dir, 'waifu2x_upconv', waifu2x_dtype, 'upconv_7', waifu2x_train_source, waifu2x_model)
                if not os.path.isfile(onnx_file):
                    raise ValueError('Data is not prepared')
                onnx.checker.check_model(onnx_file)
                waifu2x_list.append(onnx_file)
    return rife_list + tha_list + real_esrgan_list + waifu2x_list