import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import *
from os.path import join
from ezvtb_rt.engine import Engine, createMemory


class THACore:
    def __init__(self, model_dir):
        self.prepareEngines(model_dir)
        self.prepareMemories()
        self.setMemsToEngines()

    def prepareEngines(self, model_dir, engineT = Engine): #inherit and pass different engine type
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating Engines')
        self.decomposer = engineT(model_dir, 'decomposer', 1)
        self.combiner = engineT(model_dir, 'combiner', 4)
        self.morpher = engineT(model_dir, 'morpher', 4)
        self.rotator = engineT(model_dir, 'rotator', 2)
        self.editor = engineT(model_dir, 'editor', 4)

    def prepareMemories(self):
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating memories on VRAM')
        self.memories = {}
        self.memories['input_img'] = createMemory(self.decomposer.inputs[0])
        self.memories["background_layer"] = createMemory(self.decomposer.outputs[0])
        self.memories["eyebrow_layer"] = createMemory(self.decomposer.outputs[1])
        self.memories["image_prepared"] = createMemory(self.decomposer.outputs[2])

        self.memories['eyebrow_pose'] = createMemory(self.combiner.inputs[3])
        self.memories['eyebrow_image'] = createMemory(self.combiner.outputs[0])
        self.memories['morpher_decoded'] = createMemory(self.combiner.outputs[1])

        self.memories['face_pose'] = createMemory(self.morpher.inputs[2])
        self.memories['face_morphed_full'] = createMemory(self.morpher.outputs[0])
        self.memories['face_morphed_half'] = createMemory(self.morpher.outputs[1])

        self.memories['rotation_pose'] = createMemory(self.rotator.inputs[1])
        self.memories['wrapped_image'] = createMemory(self.rotator.outputs[0])
        self.memories['grid_change'] = createMemory(self.rotator.outputs[1])

        self.memories['output_img'] = createMemory(self.editor.outputs[0])
        self.memories['output_cv_img'] = createMemory(self.editor.outputs[1])

    def setMemsToEngines(self):
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Linking memories on VRAM to engine graph nodes')
        decomposer_inputs = [self.memories['input_img']]
        self.decomposer.setInputMems(decomposer_inputs)
        decomposer_outputs = [self.memories["background_layer"], self.memories["eyebrow_layer"], self.memories["image_prepared"]]
        self.decomposer.setOutputMems(decomposer_outputs)

        combiner_inputs = [self.memories['image_prepared'], self.memories["background_layer"], self.memories["eyebrow_layer"], self.memories['eyebrow_pose']]
        self.combiner.setInputMems(combiner_inputs)
        combiner_outputs = [self.memories['eyebrow_image'], self.memories['morpher_decoded']]
        self.combiner.setOutputMems(combiner_outputs)

        morpher_inputs = [self.memories['image_prepared'], self.memories['eyebrow_image'], self.memories['face_pose'], self.memories['morpher_decoded']]
        self.morpher.setInputMems(morpher_inputs)
        morpher_outputs = [self.memories['face_morphed_full'], self.memories['face_morphed_half']]
        self.morpher.setOutputMems(morpher_outputs)

        rotator_inputs = [self.memories['face_morphed_half'], self.memories['rotation_pose']]
        self.rotator.setInputMems(rotator_inputs)
        rotator_outputs = [self.memories['wrapped_image'], self.memories['grid_change']]
        self.rotator.setOutputMems(rotator_outputs)

        editor_inputs = [self.memories['face_morphed_full'], self.memories['wrapped_image'], self.memories['grid_change'], self.memories['rotation_pose']]
        self.editor.setInputMems(editor_inputs)
        editor_outputs = [self.memories['output_img'], self.memories['output_cv_img']]
        self.editor.setOutputMems(editor_outputs)


class THACoreSimple(THACore): #Simple implementation of tensorrt tha core, just for benchmarking tha's performance on given platform
    def __init__(self, model_dir):
        super().__init__(model_dir)
        # create stream
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        # Create a CUDA events
        self.finishedFetchRes = cuda.Event()
        self.finishedExec = cuda.Event()
    
    def setImage(self, img:np.ndarray):
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4)
        np.copyto(self.memories['input_img'].host, img)
        self.memories['input_img'].htod(self.updatestream)
        self.decomposer.exec(self.updatestream)
        self.updatestream.synchronize()
    def inference(self, pose:np.ndarray) -> np.ndarray: #Start inferencing given input and return result of previous frame
        
        self.outstream.wait_for_event(self.finishedExec)
        self.memories['output_cv_img'].dtoh(self.outstream)
        self.finishedFetchRes.record(self.outstream)

        np.copyto(self.memories['eyebrow_pose'].host, pose[:, :12])
        self.memories['eyebrow_pose'].htod(self.instream)
        np.copyto(self.memories['face_pose'].host, pose[:,12:12+27])
        self.memories['face_pose'].htod(self.instream)
        np.copyto(self.memories['rotation_pose'].host, pose[:,12+27:])
        self.memories['rotation_pose'].htod(self.instream)

        self.combiner.exec(self.instream)
        self.morpher.exec(self.instream)
        self.rotator.exec(self.instream)
        self.instream.wait_for_event(self.finishedFetchRes)
        self.editor.exec(self.instream)
        self.finishedExec.record(self.instream)
        
        self.finishedFetchRes.synchronize()
        return self.memories['output_cv_img'].host