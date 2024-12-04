import sys
import os
sys.path.append(os.getcwd())
import onnxruntime as ort
import onnx
from typing import List
import numpy as np

def check_merge_graph(tha_dir:str, seperable:bool):
    decomposer = onnx.load(os.path.join(tha_dir, 'decomposer.onnx'))
    decomposer = onnx.compose.add_prefix(decomposer,'decomposer_')
    combiner = onnx.load(os.path.join(tha_dir, 'combiner.onnx'))
    combiner = onnx.compose.add_prefix(combiner,'combiner_')
    morpher = onnx.load(os.path.join(tha_dir, 'morpher.onnx'))
    morpher = onnx.compose.add_prefix(morpher,'morpher_')
    rotator = onnx.load(os.path.join(tha_dir, 'rotator.onnx'))
    rotator = onnx.compose.add_prefix(rotator,'rotator_')
    editor = onnx.load(os.path.join(tha_dir, 'editor.onnx'))
    editor = onnx.compose.add_prefix(editor,'editor_')
    if not seperable:
        decoded_cut = ('combiner_/face_morpher/downsample_blocks.3/downsample_blocks.3.2/Relu_output_0', 'morpher_/face_morpher/downsample_blocks.3/downsample_blocks.3.2/Relu_output_0')
    else:
        decoded_cut = ('combiner_/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0', 'morpher_/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0')
    
    merged = onnx.compose.merge_models(rotator, editor, [("rotator_full_warped_image", 'editor_rotated_warped_image'),
                                            ("rotator_full_grid_change", 'editor_rotated_grid_change')], outputs=['editor_cv_result'])
    merged = onnx.compose.merge_models(morpher, merged, [('morpher_face_morphed_full', 'editor_morphed_image'), 
                                                         ('morpher_face_morphed_half', 'rotator_face_morphed_half')])
    merged = onnx.compose.merge_models(combiner, merged, [('combiner_eyebrow_image', 'morpher_im_morpher_crop'), 
                                                         decoded_cut])
    merged = onnx.compose.merge_models(decomposer, merged, [('decomposer_background_layer', 'combiner_eyebrow_background_layer'), 
                                                            ('decomposer_eyebrow_layer', "combiner_eyebrow_layer"),
                                                            ('decomposer_image_prepared', 'combiner_image_prepared'),
                                                            ('decomposer_image_prepared', 'morpher_image_prepared')])
    onnx.save_model(merged, os.path.join(tha_dir, 'merge.onnx'))

class THAORTCore:
    def __init__(self, tha_dir:str):
        self.tha_dir = tha_dir
        if 'fp16' in tha_dir:
            self.dtype = np.float16
        else:
            self.dtype = np.float32

        if 'seperable' in tha_dir:
            self.seperable = True
        else:
            self.seperable = False 

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
        check_merge_graph(self.tha_dir, self.seperable)

        providers = [ self.provider]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # self.decomposed_background_layer =  ort.OrtValue.ortvalue_from_shape_and_type((1,4,128,128), self.dtype, self.device)
        # self.decomposed_eyebrow_layer =  ort.OrtValue.ortvalue_from_shape_and_type((1,4,128,128), self.dtype, self.device)
        # self.image_prepared =  ort.OrtValue.ortvalue_from_shape_and_type((1,4,512,512), self.dtype, self.device)
        self.input_image =  ort.OrtValue.ortvalue_from_shape_and_type((512,512, 4), np.uint8, self.device)
        self.face_pose = ort.OrtValue.ortvalue_from_shape_and_type((1,27), np.float32, self.device)
        self.rotation_pose = ort.OrtValue.ortvalue_from_shape_and_type((1,6), np.float32, self.device)
        self.eyebrow_pose = ort.OrtValue.ortvalue_from_shape_and_type((1,12), np.float32, self.device)
        self.result_image =  ort.OrtValue.ortvalue_from_shape_and_type((512,512, 4), np.uint8, self.device)
        
        # self.decomposer_sess = ort.InferenceSession(os.path.join(tha_dir, 'decomposer.onnx'), sess_options=options, providers=providers)#, sess_options=sess_options, providers=providers)
        self.merged = ort.InferenceSession(os.path.join(tha_dir, "merge.onnx"), sess_options=options, providers=providers)

        self.binding = self.merged.io_binding()
        self.binding.bind_ortvalue_input('decomposer_input_image', self.input_image)
        self.binding.bind_ortvalue_input('combiner_eyebrow_pose', self.eyebrow_pose)
        self.binding.bind_ortvalue_input('morpher_face_pose', self.face_pose)
        self.binding.bind_ortvalue_input('rotator_rotation_pose', self.rotation_pose)
        self.binding.bind_ortvalue_input('editor_rotation_pose', self.rotation_pose)
        self.binding.bind_ortvalue_output('editor_cv_result', self.result_image)
    def update_image(self, img:np.ndarray):
        shapes = img.shape
        if len(shapes) != 3 or shapes[0]!= 512 or shapes[1] != 512 or shapes[2] != 4:
            raise ValueError('Not valid update image')
        self.input_image.update_inplace(img)

    def inference(self, poses:np.ndarray):
        
        self.eyebrow_pose.update_inplace(poses[:, :12])
        self.face_pose.update_inplace(poses[:,12:12+27])
        self.rotation_pose.update_inplace(poses[:,12+27:])

        self.merged.run_with_iobinding(self.binding)

        return self.result_image.numpy()


