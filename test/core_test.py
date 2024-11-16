import sys
import os
sys.path.append(os.getcwd())

from ezvtb_rt.core import THACore, RIFECore
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
from tqdm import tqdm
from ezvtb_rt.cv_utils import numpy_to_image_file, img_file_to_numpy

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

        self.combiner.exec(self.stream)
        self.morpher.exec(self.stream)
        self.rotator.exec(self.stream)
        self.editor.exec(self.stream)

        self.memories['output_img'].dtoh(self.stream)
        
        self.end_event.record(self.stream)
        self.stream.synchronize()
        return self.memories['output_img'].host
    
class RIFECoreSimple(RIFECore):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        # create stream
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        # Create CUDA events
        self.finishedFetchRes = cuda.Event()
        self.finishedExec = cuda.Event()


    def run(self, old_frame:np.ndarray, latest_frame:np.ndarray) -> List[np.ndarray]: # Run new and return last result at the same time

        # self.outstream.wait_for_event(self.finishedExec)
        
        # self.finishedFetchRes.record(self.outstream)

        np.copyto(self.memories['old_frame'].host, old_frame)
        self.memories['old_frame'].htod(self.instream)
        np.copyto(self.memories['latest_frame'].host, latest_frame)
        self.memories['latest_frame'].htod(self.instream)
        
        

        # self.instream.wait_for_event(self.finishedFetchRes)
        self.engine.exec(self.instream)
    
        # self.finishedExec.record(self.instream)

        # self.finishedFetchRes.synchronize()


        for i in range(self.scale -1):
            self.memories['framegen_'+str(i)].dtoh(self.instream)

        self.instream.synchronize()

        ret = []
        for i in range(self.scale -1):
            ret.append(self.memories['framegen_'+str(i)].host)
        return ret

class THAWithRIFE():
    def __init__(self, tha_model_dir, rift_model_dir):
        self.tha = THACore(tha_model_dir)
        self.rife = RIFECore(rift_model_dir, scale = -1, latest_frame=self.tha.memories['output_img'])
        
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()

        self.thaFinished = cuda.Event()
        self.rifeFinished = cuda.Event()
        self.resultFetchFinished = cuda.Event()

    def setImage(self, img:np.ndarray):
        assert(len(img.shape) == 4 and 
               img.shape[0] == 1 and 
               img.shape[1] == 4 and 
               img.shape[2] == 512 and 
               img.shape[3] == 512)
        np.copyto(self.tha.memories['input_img'].host, img)
        self.tha.memories['input_img'].htod(self.updatestream)
        self.tha.decomposer.exec(self.updatestream)
        self.updatestream.synchronize()

    def inference(self, pose:np.ndarray) -> List[np.ndarray]: # Put input and get last output

        self.outstream.wait_for_event(self.rifeFinished)
        for i in range(self.rife.scale -1):
            self.rife.memories['framegen_'+str(i)].dtoh(self.outstream)

        cuda.memcpy_dtod_async(self.rife.memories['old_frame'].device, self.tha.memories['output_img'].device, 
                               self.rife.memories['old_frame'].host.nbytes, self.outstream)
        self.resultFetchFinished.record(self.outstream)

        np.copyto(self.tha.memories['eyebrow_pose'].host, pose[:, :12])
        self.tha.memories['eyebrow_pose'].htod(self.instream)
        np.copyto(self.tha.memories['face_pose'].host, pose[:,12:12+27])
        self.tha.memories['face_pose'].htod(self.instream)
        np.copyto(self.tha.memories['rotation_pose'].host, pose[:,12+27:])
        self.tha.memories['rotation_pose'].htod(self.instream)
        # self.instream.wait_for_event(self.thaFinished)

        self.tha.combiner.exec(self.instream)
        self.tha.morpher.exec(self.instream)
        self.tha.rotator.exec(self.instream)
        
        self.tha.editor.exec(self.instream)

        self.instream.wait_for_event(self.resultFetchFinished)
        self.rife.engine.exec(self.instream)
        self.rifeFinished.record(self.instream)

        self.resultFetchFinished.synchronize()

        ret = []
        for i in range(self.rife.scale -1):
            ret.append(self.rife.memories['framegen_'+str(i)].host)

        ret.append(self.rife.memories['old_frame'])
        return ret






def THATestPerf():
    core = THACoreSimple('./data/tha3/seperable/fp16')
    cuda.start_profiler()
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
    cuda.stop_profiler()


def RIFETestPerf():
    
    img1 = np.random.rand(1,4,512,512).astype(np.float16)
    img2 = np.random.rand(1,4,512,512).astype(np.float16)
    cuda.start_profiler()
    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x2')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x3')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4')
    for i in tqdm(range(1000)):
        ret = core.run(img1, img2)

    cuda.stop_profiler()

def THAWithRifePerf():
    core = THAWithRIFE('./data/tha3/seperable/fp32', './data/rife_lite_v4_25/rife_x4', 'rife_512')
    core.setImage(np.random.rand(1,4,512,512).astype(np.float16))
    for i in tqdm(range(1000)):
        core.inference(np.random.rand(1,45).astype(np.float16))

def THATestShow():
    core = THACoreSimple('./data/tha3/seperable/fp16')
    core.setImage( img_file_to_numpy('./test/data/tha/base.png'))
    pose = np.zeros((1,45)).astype(np.float16)
    for i in range(10):
        pose[:,:12] = np.random.rand(1,12) * 2.0 - 1.0
        pose[:,-6:] = np.random.rand(1,6) * 2.0 - 1.0
        ret = core.inference(pose)
        numpy_to_image_file(ret, f'./test/data/tha/base_seperable_fp16_{i}.png')
    
    core = THACoreSimple('./data/tha3/seperable/fp32')
    core.setImage( img_file_to_numpy('./test/data/tha/base.png'))
    pose = np.zeros((1,45)).astype(np.float16)
    for i in range(10):
        pose[:,:12] = np.random.rand(1,12) * 2.0 - 1.0
        pose[:,-6:] = np.random.rand(1,6) * 2.0 - 1.0
        ret = core.inference(pose)
        numpy_to_image_file(ret, f'./test/data/tha/base_seperable_fp32_{i}.png')

    core = THACoreSimple('./data/tha3/standard/fp32')
    core.setImage( img_file_to_numpy('./test/data/tha/base.png'))
    pose = np.zeros((1,45)).astype(np.float16)
    for i in range(10):
        pose[:,:12] = np.random.rand(1,12) * 2.0 - 1.0
        pose[:,-6:] = np.random.rand(1,6) * 2.0 - 1.0
        ret = core.inference(pose)
        numpy_to_image_file(ret, f'./test/data/tha/base_standard_fp32_{i}.png')

    core = THACoreSimple('./data/tha3/standard/fp16')
    core.setImage( img_file_to_numpy('./test/data/tha/base.png'))
    pose = np.zeros((1,45)).astype(np.float16)
    for i in range(10):
        pose[:,:12] = np.random.rand(1,12) * 2.0 - 1.0
        pose[:,-6:] = np.random.rand(1,6) * 2.0 - 1.0
        ret = core.inference(pose)
        numpy_to_image_file(ret, f'./test/data/tha/base_standard_fp16_{i}.png')

def RIFETestShow():
    img_0 = img_file_to_numpy('./test/rife/data/0.png')
    img_1 = img_file_to_numpy('./test/rife/data/1.png')
    core = RIFECoreSimple('./data/rife_lite_v4_25/rife_x4')
    core.run(img_0, img_1)
    res = core.run(img_0, img_1)
    numpy_to_image_file(res[0]*2 -1, './test/data/0_25.png')
    numpy_to_image_file(res[1]*2 -1, './test/data/0_50.png')
    numpy_to_image_file(res[2]*2 -1, './test/data/0_75.png')
    numpy_to_image_file(res[3]*2 -1, './test/data/0_80.png')
    numpy_to_image_file(res[4]*2 -1, './test/data/0_90.png')


if __name__ == "__main__":
    # THAWithRifePerf()
    # THATestPerf()
    RIFETestPerf()
    # THATestShow()
    # RIFETestShow()