from trt_utils import *
from tqdm import tqdm
import pycuda.autoinit

from os.path import join
def convert(sourceDir, dstDir, componentName, dataType):
    engine_seri = build_engine(join(sourceDir, componentName+'.onnx'), dataType)
    save_engine(engine_seri, join(dstDir, componentName+'.trt'))


# convert('./data/tha3/seperable/fp16/', './data/tha3/seperable/fp16/','combiner', 'fp16')
# convert('./data/tha3/seperable/fp16/', './data/tha3/seperable/fp16/','decomposer', 'fp16')
# convert('./data/tha3/seperable/fp16/', './data/tha3/seperable/fp16/','editor', 'fp16')
# convert('./data/tha3/seperable/fp16/', './data/tha3/seperable/fp16/','morpher', 'fp16')
# convert('./data/tha3/seperable/fp16/', './data/tha3/seperable/fp16/','rotator', 'fp16')

def fullbench(model_dir, dtype, iters = 100000):
    decomposer_engine = load_engine(join(model_dir, 'decomposer.trt'))
    decomposer_proc = Processor(decomposer_engine, 4)
    
    combiner_engine = load_engine(join(model_dir, 'combiner.trt'))
    combiner_proc = Processor(combiner_engine, 4)
    
    morpher_engine = load_engine(join(model_dir, 'morpher.trt'))
    morpher_proc = Processor(morpher_engine, 4)
    
    rotator_engine = load_engine(join(model_dir, 'rotator.trt'))
    rotator_proc = Processor(rotator_engine, 2)
    
    editor_engine = load_engine(join(model_dir, 'editor.trt'))
    editor_proc = Processor(editor_engine, 2)

    for i in range(100): #preheat
        combiner_proc.kickoff()
        morpher_proc.kickoff()
        rotator_proc.kickoff()
        editor_proc.kickoff()
    from time import time
    t1 = time()
    for i in tqdm(range(iters)):
        if i % 15 == 0:
            combiner_proc.kickoff()
        morpher_proc.kickoff()
        rotator_proc.kickoff()
        editor_proc.kickoff()
    print(time() - t1)
fullbench('./data/tha3/seperable/fp16/', np.float16, 1000)