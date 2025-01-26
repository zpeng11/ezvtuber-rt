conda install -y nvidia/label/cuda-12.6.3::cuda-nvcc-dev_win-64
conda install conda-forge::pycuda onnx onnxruntime-directml turbojpeg tqdm opencv-python
python -m pip install --upgrade pip wheel
python -m pip install nvidia-cudnn-cu12
pip install tensorrt_cu12_libs==10.6.0 tensorrt_cu12_bindings==10.6.0 tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com