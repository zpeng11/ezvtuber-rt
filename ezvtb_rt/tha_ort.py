import os
import onnxruntime as ort
import onnx
import numpy as np
from typing import List

def merge_graph_all(tha_dir:str, seperable:bool):
    try:
        onnx.checker.check_model(os.path.join(tha_dir, 'merge_all.onnx'))
        return
    except:
        pass
    #merge all models into one
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
    onnx.save_model(merged, os.path.join(tha_dir, 'merge_all.onnx'))

def merge_graph(tha_dir:str, seperable:bool, use_eyebrow:bool = True):
    try:
        onnx.checker.check_model(os.path.join(tha_dir, 'merge.onnx' if use_eyebrow else 'merge_no_eyebrow.onnx'))
        return
    except:
        pass
    #Merge models except for decomposer
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
    
    if use_eyebrow:
        merged = onnx.compose.merge_models(combiner, merged, [('combiner_eyebrow_image', 'morpher_im_morpher_crop'), 
                                                         decoded_cut])
    onnx.save_model(merged, os.path.join(tha_dir, 'merge.onnx' if use_eyebrow else 'merge_no_eyebrow.onnx'))

class THAORT:
    def __init__(self, tha_dir:str, use_eyebrow:bool = True):
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
        
        self.use_eyebrow = use_eyebrow

        merge_graph(self.tha_dir, self.seperable, use_eyebrow)

        providers = [ self.provider]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if use_eyebrow:
            self.decomposed_background_layer =  ort.OrtValue.ortvalue_from_shape_and_type((1,4,128,128), self.dtype, self.device)
            self.decomposed_eyebrow_layer =  ort.OrtValue.ortvalue_from_shape_and_type((1,4,128,128), self.dtype, self.device)
            self.eyebrow_pose = ort.OrtValue.ortvalue_from_shape_and_type((1,12), np.float32, self.device)
        else:
            self.combiner_eyebrow_image = ort.OrtValue.ortvalue_from_shape_and_type((1,4,192,192), self.dtype, self.device)
            self.combiner_decode_cut = ort.OrtValue.ortvalue_from_shape_and_type((1,512,24,24), self.dtype, self.device)
        self.image_prepared =  ort.OrtValue.ortvalue_from_shape_and_type((1,4,512,512), self.dtype, self.device)
        self.face_pose = ort.OrtValue.ortvalue_from_shape_and_type((1,27), np.float32, self.device)
        self.rotation_pose = ort.OrtValue.ortvalue_from_shape_and_type((1,6), np.float32, self.device)
        self.result_image =  ort.OrtValue.ortvalue_from_shape_and_type((512,512, 4), np.uint8, self.device)
        
        self.decomposer = ort.InferenceSession(os.path.join(tha_dir, 'decomposer.onnx'), sess_options=options, providers=providers)
        if not use_eyebrow:
            self.combiner = ort.InferenceSession(os.path.join(tha_dir, 'combiner.onnx'), sess_options=options, providers=providers)
        self.merged = ort.InferenceSession(os.path.join(tha_dir, "merge.onnx" if use_eyebrow else 'merge_no_eyebrow.onnx'), sess_options=options, providers=providers)

        self.binding = self.merged.io_binding()
        if use_eyebrow:
            self.binding.bind_ortvalue_input('combiner_eyebrow_background_layer', self.decomposed_background_layer)
            self.binding.bind_ortvalue_input('combiner_eyebrow_layer', self.decomposed_eyebrow_layer)
            self.binding.bind_ortvalue_input('combiner_eyebrow_pose', self.eyebrow_pose)
            self.binding.bind_ortvalue_input('combiner_image_prepared', self.image_prepared)
        else: #no eyebrow
            self.binding.bind_ortvalue_input('morpher_im_morpher_crop', self.combiner_eyebrow_image)
            if not self.seperable:
                decoded_cut_name = 'morpher_/face_morpher/downsample_blocks.3/downsample_blocks.3.2/Relu_output_0'
            else:
                decoded_cut_name = 'morpher_/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0'
            self.binding.bind_ortvalue_input(decoded_cut_name, self.combiner_decode_cut)
        self.binding.bind_ortvalue_input('morpher_image_prepared', self.image_prepared)
        self.binding.bind_ortvalue_input('morpher_face_pose', self.face_pose)
        self.binding.bind_ortvalue_input('rotator_rotation_pose', self.rotation_pose)
        self.binding.bind_ortvalue_input('editor_rotation_pose', self.rotation_pose)
        self.binding.bind_ortvalue_output('editor_cv_result', self.result_image)

    def update_image(self, img:np.ndarray):
        shapes = img.shape
        if len(shapes) != 3 or shapes[0]!= 512 or shapes[1] != 512 or shapes[2] != 4:
            raise ValueError('Not valid update image')
        decomposed = self.decomposer.run(None, {'input_image':img})
        self.image_prepared.update_inplace(decomposed[2])
        if self.use_eyebrow:
            self.decomposed_background_layer.update_inplace(decomposed[0])
            self.decomposed_eyebrow_layer.update_inplace(decomposed[1])
        else:
            combined = self.combiner.run(None, {
                'image_prepared':decomposed[2],
                'eyebrow_background_layer': decomposed[0],
                'eyebrow_layer':decomposed[1],
                'eyebrow_pose':np.zeros((1,12), dtype=np.float32)
            })
            self.combiner_eyebrow_image.update_inplace(combined[0])
            self.combiner_decode_cut.update_inplace(combined[1])


    def inference(self, poses:np.ndarray) -> List[np.ndarray]:
        if self.use_eyebrow:
            self.eyebrow_pose.update_inplace(poses[:, :12])
        self.face_pose.update_inplace(poses[:,12:12+27])
        self.rotation_pose.update_inplace(poses[:,12+27:])

        self.merged.run_with_iobinding(self.binding)

        return [self.result_image.numpy()]


class THAORTNonDefault:
    #Interesting bug in onnxruntime that ortValue with dml only support default device (device=0), 
    #Which means when using a nondefault ORT device, we can not use any vram cache but have to merge graph to reduce passage through pcie boundary
    def __init__(self, tha_dir:str, device_id:int, use_eyebrow = True):
        # if device_id == 0:
            # raise ValueError('Use the default version for this device because that is faster')
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
        
        self.use_eyebrow = use_eyebrow
        merge_graph(self.tha_dir, self.seperable, use_eyebrow)

        providers = [ self.provider]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider_options = [{'device_id':device_id}]

        self.decomposer = ort.InferenceSession(os.path.join(tha_dir,  "decomposer.onnx"), sess_options=options, providers=providers, provider_options=provider_options)
        self.combiner = ort.InferenceSession(os.path.join(tha_dir,  "combiner.onnx"), sess_options=options, providers=providers, provider_options=provider_options)
        self.merged = ort.InferenceSession(os.path.join(tha_dir,  "merge.onnx" if use_eyebrow else 'merge_no_eyebrow.onnx'), sess_options=options, providers=providers, provider_options=provider_options)

        if not self.seperable:
            self.decoded_cut_name = 'morpher_/face_morpher/downsample_blocks.3/downsample_blocks.3.2/Relu_output_0'
        else:
            self.decoded_cut_name = 'morpher_/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0'

    def update_image(self, img:np.ndarray):
        self.decomposed = self.decomposer.run(None, {
            'input_image':img
        })
        self.combined = self.combiner.run(None, {
            'image_prepared': self.decomposed[2],
            'eyebrow_background_layer':self.decomposed[0],
            'eyebrow_layer':self.decomposed[1],
            'eyebrow_pose':np.zeros((1,12),dtype=np.float32)
        })

    def inference(self, poses:np.ndarray) -> List[np.ndarray]:
        if self.use_eyebrow:
            return self.merged.run(None, {
                'combiner_image_prepared' : self.decomposed[2],
                'combiner_eyebrow_background_layer' : self.decomposed[0],
                'combiner_eyebrow_layer' : self.decomposed[1],
                'combiner_eyebrow_pose': poses[:, :12],
                'morpher_image_prepared' :self.decomposed[2],
                'morpher_face_pose':poses[:,12:12+27],
                'rotator_rotation_pose':poses[:,12+27:],
                'editor_rotation_pose':poses[:,12+27:],
            })
        else:
            return self.merged.run(None, {
                'morpher_image_prepared' : self.decomposed[2],
                'morpher_im_morpher_crop': self.combined[0],
                'morpher_face_pose':poses[:,12:12+27],
                self.decoded_cut_name:self.combined[1],
                'rotator_rotation_pose':poses[:,12+27:],
                'editor_rotation_pose':poses[:,12+27:],
            })