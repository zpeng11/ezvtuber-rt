import os
from datetime import datetime
import onnxruntime as ort


def _ts():
    return datetime.now().strftime('[%m/%d/%Y-%H:%M:%S]')


def createORTSession(model_path: str, device_id: int = 0):
    """创建 ONNX Runtime 推理会话。会输出加载流程和当前使用的模型文件名（与 TRT 加载时的输出风格对应）。"""
    filename = os.path.basename(model_path)
    print(f'{_ts()} [ORT] Loading ONNX model from path {model_path}...')
    providers = ['DmlExecutionProvider']
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    provider_options = [{'device_id': device_id, "execution_mode": "parallel", "arena_extend_strategy": "kSameAsRequested"}]
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers, provider_options=provider_options)
    print(f'{_ts()} [ORT] Completed loading session: {filename}')
    return session