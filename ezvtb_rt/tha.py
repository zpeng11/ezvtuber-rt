import sys
import os
sys.path.append(os.getcwd())
from ezvtb_rt.trt_utils import *
from os.path import join
from ezvtb_rt.engine import Engine, createMemory
from collections import OrderedDict


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

#memory management protector of VRAM
class VRAMMem(object):
    def __init__(self, nbytes:int):
        self.device = cuda.mem_alloc(nbytes)

    def __str__(self):
        return "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    def __del__(self):
        self.device.free()

class THACoreCached(THACore): #Cached implementation of tensorrt tha core
    def __init__(self, model_dir, vram_cache_size:float = 1):
        super().__init__(model_dir)
        self.cache = OrderedDict()
        self.cached_kbytes = 0
        self.max_cached_kbytes = int(vram_cache_size * 1024 * 1024)
        self.morpher_cache_size = self.memories['face_morphed_full'].host.nbytes + self.memories['face_morphed_half'].host.nbytes
        self.combiner_cache_size = self.memories['eyebrow_image'].host.nbytes + self.memories['morpher_decoded'].host.nbytes
        self.hits = 0
        self.miss = 0
        # create stream
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.cachestream = cuda.Stream()
        # Create a CUDA events
        self.finishedMorpher = cuda.Event()
        self.finishedCombiner = cuda.Event()
    
    def setImage(self, img:np.ndarray):
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4)
        np.copyto(self.memories['input_img'].host, img)
        self.memories['input_img'].htod(self.updatestream)
        self.decomposer.exec(self.updatestream)
        self.updatestream.synchronize()

    def inference(self, pose:np.ndarray) -> np.ndarray: #This inference is running in a synchronized way
        eyebrow_pose = pose[:, :12]
        face_pose = pose[:,12:12+27]
        rotation_pose = pose[:,12+27:]

        np.copyto(self.memories['rotation_pose'].host, rotation_pose)
        self.memories['rotation_pose'].htod(self.instream)

        morpher_hash = hash(str(pose[:,:12+27]))
        morpher_cached = self.cache.get(morpher_hash)
        combiner_hash = hash(str(pose[:,:12]))
        combiner_cached = self.cache.get(combiner_hash)

        self.cachestream.synchronize()
        if(morpher_cached is not None):
            self.hits += 1
            self.cache.move_to_end(morpher_hash)
            cuda.memcpy_dtod_async(self.memories['face_morphed_full'].device, morpher_cached[0].device, 
                                   self.memories['face_morphed_full'].host.nbytes, self.instream)
            cuda.memcpy_dtod_async(self.memories['face_morphed_half'].device, morpher_cached[1].device, 
                                   self.memories['face_morphed_half'].host.nbytes, self.instream)
            self.rotator.exec(self.instream)
            self.editor.exec(self.instream)
            self.memories['output_cv_img'].dtoh(self.instream)
        elif(combiner_cached is not None):
            self.hits += 1
            self.cache.move_to_end(combiner_hash)
            #prepare input
            cuda.memcpy_dtod_async(self.memories['eyebrow_image'].device, combiner_cached[0].device, 
                                   self.memories['eyebrow_image'].host.nbytes, self.instream)
            cuda.memcpy_dtod_async(self.memories['morpher_decoded'].device, combiner_cached[1].device, 
                                   self.memories['morpher_decoded'].host.nbytes, self.instream)
            np.copyto(self.memories['face_pose'].host, face_pose)
            self.memories['face_pose'].htod(self.instream)
            
            #Execute morpher
            self.morpher.exec(self.instream)
            self.finishedMorpher.record(self.instream)

            #cache morpher result
            self.cachestream.wait_for_event(self.finishedMorpher)
            face_morphed_full_cached = VRAMMem(self.memories['face_morphed_full'].host.nbytes)
            face_morphed_half_cached = VRAMMem(self.memories['face_morphed_half'].host.nbytes)
            cuda.memcpy_dtod_async(face_morphed_full_cached.device, self.memories['face_morphed_full'].device, 
                                   self.memories['face_morphed_full'].host.nbytes, self.cachestream)
            cuda.memcpy_dtod_async(face_morphed_half_cached.device, self.memories['face_morphed_half'].device, 
                                   self.memories['face_morphed_half'].host.nbytes, self.cachestream)
            
            #Execute the rest
            self.rotator.exec(self.instream)
            self.editor.exec(self.instream)
            self.memories['output_cv_img'].dtoh(self.instream)

            #save morpher cache
            self.cache[morpher_hash] = (face_morphed_full_cached, face_morphed_half_cached, self.morpher_cache_size)
            self.cached_kbytes += (self.morpher_cache_size)//1024
            while(self.cached_kbytes > self.max_cached_kbytes):
                poped = self.cache.popitem(last=False)
                self.cached_kbytes -= poped[1][2] 
        else:
            self.miss += 1
            #prepare input
            np.copyto(self.memories['face_pose'].host, face_pose)
            self.memories['face_pose'].htod(self.instream)
            np.copyto(self.memories['eyebrow_pose'].host, eyebrow_pose)
            self.memories['eyebrow_pose'].htod(self.instream)

            #execute combiner
            self.combiner.exec(self.instream)
            self.finishedCombiner.record(self.instream)

            #cache morpher result
            self.cachestream.wait_for_event(self.finishedCombiner)
            eyebrow_image_cached = VRAMMem(self.memories['eyebrow_image'].host.nbytes)
            morpher_decoded_cached = VRAMMem(self.memories['morpher_decoded'].host.nbytes)
            cuda.memcpy_dtod_async(eyebrow_image_cached.device, self.memories['eyebrow_image'].device, 
                                   self.memories['eyebrow_image'].host.nbytes, self.cachestream)
            cuda.memcpy_dtod_async(morpher_decoded_cached.device, self.memories['morpher_decoded'].device, 
                                   self.memories['morpher_decoded'].host.nbytes, self.cachestream)
            
            #execute morpher
            self.morpher.exec(self.instream)
            self.finishedMorpher.record(self.instream)
            
            #cache morpher result
            self.cachestream.wait_for_event(self.finishedMorpher)
            face_morphed_full_cached = VRAMMem(self.memories['face_morphed_full'].host.nbytes)
            face_morphed_half_cached = VRAMMem(self.memories['face_morphed_half'].host.nbytes)
            cuda.memcpy_dtod_async(face_morphed_full_cached.device, self.memories['face_morphed_full'].device, 
                                   self.memories['face_morphed_full'].host.nbytes, self.cachestream)
            cuda.memcpy_dtod_async(face_morphed_half_cached.device, self.memories['face_morphed_half'].device, 
                                   self.memories['face_morphed_half'].host.nbytes, self.cachestream)
            
            #execute the rest
            self.rotator.exec(self.instream)
            self.editor.exec(self.instream)
            self.memories['output_cv_img'].dtoh(self.instream)

            #save caches
            self.cache[combiner_hash] = (eyebrow_image_cached, morpher_decoded_cached, self.combiner_cache_size)
            self.cached_kbytes += (self.combiner_cache_size)//1024
            self.cache[morpher_hash] = (face_morphed_full_cached, face_morphed_half_cached, self.morpher_cache_size)
            self.cached_kbytes += (self.morpher_cache_size)//1024
            while(self.cached_kbytes > self.max_cached_kbytes):
                poped = self.cache.popitem(last=False)
                self.cached_kbytes -= poped[1][2] 

        self.instream.synchronize()
        return self.memories['output_cv_img'].host