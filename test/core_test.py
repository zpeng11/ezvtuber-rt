import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.core import THACore, RIFECore
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm

class THACoreSimple(THACore):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)
    
    def setImage(self, img:np.ndarray):
        assert(len(img.shape) == 4 and 
               img.shape[0] == 1 and 
               img.shape[1] == 4 and 
               img.shape[2] == 512 and 
               img.shape[3] == 512)
        np.copyto(self.memories['input_img'].host, img)
        self.memories['input_img'].htod(self.stream)
        self.decomposer.exec(self.stream)
        self.stream.synchronize()
    def inference(self, pose:np.ndarray):
        self.start_event.record(self.stream)

        np.copyto(self.memories['eyebrow_pose'].host, pose[:, :12])
        self.memories['eyebrow_pose'].htod(self.stream)
        np.copyto(self.memories['face_pose'].host, pose[:,12:12+27])
        self.memories['face_pose'].htod(self.stream)
        np.copyto(self.memories['rotation_pose'].host, pose[:,12+27:])
        self.memories['rotation_pose'].htod(self.stream)

        self.decomposer.exec(self.stream)
        self.combiner.exec(self.stream)
        self.morpher.exec(self.stream)
        self.rotator.exec(self.stream)
        self.editor.exec(self.stream)

        self.memories['output_img'].dtoh(self.stream)
        
        self.end_event.record(self.stream)
        self.stream.synchronize()
        return self.memories['output_img'].host
    
class RIFECoreSimple(RIFECore):
    def __init__(self, model_dir, model_component):
        super().__init__(model_dir, model_component)
        # create stream
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        # Create CUDA events
        self.finishedFetchRes = cuda.Event()
        self.finishedExec = cuda.Event()


    def run(self, old_frame:np.ndarray, latest_frame:np.ndarray) -> List[np.ndarray]: # Run new and return last result at the same time

        np.copyto(self.memories['old_frame'].host, old_frame)
        self.memories['old_frame'].htod(self.instream)
        np.copyto(self.memories['latest_frame'].host, latest_frame)
        self.memories['latest_frame'].htod(self.instream)

        self.outstream.wait_for_event(self.finishedExec)
        for i in range(self.scale -1):
            self.memories['framegen_'+str(i)].dtoh(self.outstream)
        self.finishedFetchRes.record(self.outstream)

        self.instream.wait_for_event(self.finishedFetchRes)
        self.engine.exec(self.instream)
    
        self.finishedExec.record(self.instream)


        ret = []
        for i in range(self.scale -1):
            ret.append(self.memories['framegen_'+str(i)].host)
        # ret.append(self.memories['latest_frame'].host.copy())
        return ret



def THATest():
    core = THACoreSimple('./data/tha3/seperable/fp16')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float16))

    core = THACoreSimple('./data/tha3/seperable/fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float32))

    core = THACoreSimple('./data/tha3/standard/fp16')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float16))

    core = THACoreSimple('./data/tha3/standard/fp32')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float32))
    for i in tqdm(range(1000)):
        ret = core.inference(np.random.rand(1, 45).astype(np.float32))


def RIFETest():
    
    img1 = np.random.rand(1,3,512,512).astype(np.float16)
    img2 = np.random.rand(1,3,512,512).astype(np.float16)
    cuda.start_profiler()
    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2', 'rife_512')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2', 'rife_384')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3', 'rife_512')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3', 'rife_384')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4', 'rife_512')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4', 'rife_384')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)
    cuda.stop_profiler()

if __name__ == "__main__":
    THATest()
    RIFETest()