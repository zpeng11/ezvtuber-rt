import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import *
from os.path import join
import numpy as np
from abc import ABC

class NodeInfo:
    def __init__(self, name:str, shape:List[int], dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def __str__(self) -> str:
        return self.name+": "+str(self.shape) + " "+str(self.dtype)
    def __repr__(self):
        return self.__str__()
    
def createMemory(nodeInfo : NodeInfo):
    shape = nodeInfo.shape
    dtype = nodeInfo.dtype
    host_mem = cuda.pagelocked_empty(shape, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    return HostDeviceMem(host_mem, device_mem)

class Engine:
    def __init__(self, model_dir:str, model_component:str, n_inputs:int):
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {join(model_dir, model_component)}')
        dtype = 'fp32' if 'fp32' in model_dir else 'fp16'
        self.engine = get_trt_engine(model_dir, model_component, dtype)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')

        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        self.context = self.engine.create_execution_context()
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed inference context')
        
        self.inputs = []
        for i in range(n_inputs):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.inputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.inputs))

        self.outputs = []
        for i in range(n_inputs, self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.outputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.outputs))

    def setInputMems(self, inputMems:List[HostDeviceMem]): # inputs pass by reference
        assert(len(inputMems) == len(self.inputs))
        for i in range(len(inputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.inputs[i].name)
            self.context.set_tensor_address(self.inputs[i].name, int(inputMems[i].device)) # Use this setup without binding for v3

    def setOutputMems(self, outputMems:List[HostDeviceMem]): # outputs pass by reference
        assert(len(outputMems) == len(self.outputs))
        for i in range(len(outputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.outputs[i].name)
            self.context.set_tensor_address(self.outputs[i].name, int(outputMems[i].device)) # Use this setup without binding for v3
    def exec(self, stream):
        self.context.execute_async_v3(stream.handle)

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

    def setMemsToEngines(self):
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Linking memories on VRAM to engine graph nodes')
        decomposer_inputs = [self.memories['input_img']]
        self.decomposer.setInputMems(decomposer_inputs)
        decomposer_outputs = [self.memories["background_layer"], self.memories["eyebrow_layer"]]
        self.decomposer.setOutputMems(decomposer_outputs)

        combiner_inputs = [self.memories['input_img'], self.memories["background_layer"], self.memories["eyebrow_layer"], self.memories['eyebrow_pose']]
        self.combiner.setInputMems(combiner_inputs)
        combiner_outputs = [self.memories['eyebrow_image'], self.memories['morpher_decoded']]
        self.combiner.setOutputMems(combiner_outputs)

        morpher_inputs = [self.memories['input_img'], self.memories['eyebrow_image'], self.memories['face_pose'], self.memories['morpher_decoded']]
        self.morpher.setInputMems(morpher_inputs)
        morpher_outputs = [self.memories['face_morphed_full'], self.memories['face_morphed_half']]
        self.morpher.setOutputMems(morpher_outputs)

        rotator_inputs = [self.memories['face_morphed_half'], self.memories['rotation_pose']]
        self.rotator.setInputMems(rotator_inputs)
        rotator_outputs = [self.memories['wrapped_image'], self.memories['grid_change']]
        self.rotator.setOutputMems(rotator_outputs)

        editor_inputs = [self.memories['face_morphed_full'], self.memories['wrapped_image'], self.memories['grid_change'], self.memories['rotation_pose']]
        self.editor.setInputMems(editor_inputs)
        editor_outputs = [self.memories['output_img']]
        self.editor.setOutputMems(editor_outputs)

class RIFECore:
    def __init__(self, model_dir:str, model_component:str, scale:int = -1, latest_frame:HostDeviceMem = None):
        if scale < 2:
            if 'x2' in model_dir:
                self.scale = 2
            elif 'x3' in model_dir:
                self.scale = 3
            elif 'x4' in model_dir:
                self.scale = 4
            else:
                raise ValueError('can not determine scale')
        else:
            self.scale = scale
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'RIFE scale {self.scale}')
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating RIFE engine')
        self.prepareEngines(model_dir, model_component)
        self.prepareMemories(latest_frame)
        self.setMemsToEngines()

    def prepareEngines(self, model_dir:str, model_component:str, engineT = Engine): #inherit and pass different engine type
        self.engine = engineT(model_dir, model_component, 2)
    def prepareMemories(self, latest_frame:HostDeviceMem): 
        self.memories = {}
        self.memories['old_frame'] = createMemory(self.engine.inputs[0])
        if latest_frame is None:
            self.memories['latest_frame'] = createMemory(self.engine.inputs[1])
        else:
            self.memories['latest_frame'] = latest_frame
        for i in range(self.scale - 1):
            self.memories['framegen_'+str(i)] = createMemory(self.engine.outputs[i])
    def setMemsToEngines(self):
        self.engine.setInputMems([self.memories['old_frame'], self.memories['latest_frame']])
        outputs = [self.memories['framegen_'+str(i)] for i in range(self.scale - 1)]
        self.engine.setOutputMems(outputs)


    
    

