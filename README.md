# eztuber-rt
> 不用3080，也不用80%占用，3060也能开着模型玩黑猴
> 
> Inspired and motivated by [THA3](https://github.com/pkhungurn/talking-head-anime-3-demo), [EeasyVtuber](https://github.com/yuyuyzl/EasyVtuber) and [RIFE](https://github.com/hzwer/ECCV2022-RIFE). Rapid implementation powered by Nvidia [TensorRT](https://github.com/NVIDIA/TensorRT) inference framework. Output 4x super-resolution powered by [AnimeVideo-v3 model](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md) from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), and 2x super-resolution powered by [UpConv7 model](https://github.com/nagadomi/nunif/releases/tag/0.0.0) from [waifu2x project](https://github.com/nagadomi/waifu2x)

## Purpose
The primary goal of this project is to provide ONNX models and integrated code for model usage for the [EasyVtuber](https://github.com/zpeng11/EasyVtuber) project. All functionalities are encapsulated within the Core module. Please refer to [interface](https://github.com/zpeng11/EasyVtuber/blob/main/ezvtb_rt_interface.py) to create an interface file for usage. Contributions and further development are welcome.

本项目主旨在为 [EasyVtuber](https://github.com/zpeng11/EasyVtuber) 项目提供onnx模型和模型使用的代码整合，以`Core`核心包装所有功能实现，请参考 [此接口](https://github.com/zpeng11/EasyVtuber/blob/main/ezvtb_rt_interface.py) 创建接口文件使用。欢迎二次开发。

## Minimum Requeirement
### Graphic Card
Most modern discrete gaming level graphic card! AMD RX580 and up, Intel Arc a750 and up, Nvidia GTX1650 and up. (Minimum standard is meet by reaching 25fps debug output after interpolation)

大部分现代游戏级独显均可在插帧无超分无缓存下跑到25fps满足最小启动需求。如AMD RX580, 因特尔ARC A750, 英伟达 GTX1650。
## Key Improvements

### GPU Interface
Supporting DirectML for AMD and Intel GPU, cuda/TensorRT accelerate for Nvidia. DML implementation does not have good support on memory management due to ORT Python limitation.TensorRT supports Nvidia Turing microarchitecture graphic cards and above. Any gaming card higher than GeForce 16/20 series can get benefit from TensorRT and have advance memory management through PyCuda. For example all my tests are done on a GTX1660SUPER without any issue. Better gpu would give higher framerate, allow using higher percision model, and lower resource consumption.

A卡I卡使用DirectML, 有性能但不多，而且python的ort接口无法避免跨设备拷贝有性能损失。n卡可使用pycuda和TensorRT。TensorRT加速支持16和20系以上的英伟达显卡，配合cuda接口性能有跨越式提升(非严谨测试相同模型相比cuDNN实现提升约30%)，更好的显卡可以输出更高原生帧率，使用精度更高的模型和更少的系统占用。

### Static Models
Made static ONNX format model from THA3, RIFE, Waifu2x, and Real-ESRGAN's prior dynamic PyTorch implementations, which allows graph level optimization and solid inference framework support. Removed dependency from PyTorch. Feel free to take these onnx models for your own inference framework. Some models are generated from my [playground](https://github.com/zpeng11/talking-head-anime-3-demo/tree/start_testing). You can diff with main to see my work on it.

深度拆解 THA3, RIFE, Waifu2x, 和 Real-ESRGAN 模型并完成ONNX静态化。允许图优化及推理框架使用。去除 Pytorch 拥有更简单环境依赖和更快启动时间。如感兴趣请参考 [playground](https://github.com/zpeng11/talking-head-anime-3-demo/tree/start_testing) 了解静态化流程。

### TensrRT
Powered by Nvidia TensorRT which unlocks the full computation potential of the GPU and acelerates using FP16, BF16, and TF32.

使用TenosorRT框架进行推理，可使用多种精度进行加速。

### DirectML
Currently supporting AMD and Intel GPU by DirectML execution provider of OnnxRuntime. Due to Python API limitation, this method is not well optimized.

A卡和I卡使用OnnxRuntime 提供的DirectML支持，可用但因为Python接口不完善有诸多限制，此实现并非本项目主要实现方向，仅提供入门支持，请自行斟酌。

### Cache
Updated cache structure, provide VRAM+RAM solutions for caching results effectively lower down GPU resource comsumption. Use SIMD library TurboJPEG to save space.

实现显存，内存缓存器，有效减少gpu计算和显存占用。使用SIMD支持的TurboJPEG库实现快速图像压缩解压减少储存压力。


### RIFE
Introduced RIFE model to perform frame interpolation between 2 native frame comes out from THA. Resource consumption of RIFE is significantly less than THA. Just to noticed frame interpolation would bring system delay as a tradeoff. Provided up x4 times interpolation. This feature is crucial for low-end gpus like my 1660s. Limited by RIFE, interpolation up to x4 times, which brings 6fps native frame gen to 24fps experience.

使用RIFE模型进行实时插帧，带来更高帧数更少资源消耗，作为交换引入了一定系统延迟，请自行把控。低端卡福音！受制于RIFE模型设计，最高4倍插帧，等同于将6帧原生画面拉进24帧可用级别，即使高端卡也受益获得更低显卡占用，无需双卡双机。

### Super Resolution
Brings 512x512 pixels original output to 1024x1024 pixels 2x full body SR using waifu2x model, or brings the face+shoulder+half body 256x256 pixels to 1024x1024 pixels SR using Real-ESRGAN's AnimeVideo-v3 model. Please choose according to your hardware

提供两种模型使用，将全身像素2倍超分输出1024x1024的Waifu2x实现，以及将半身的256x256像素4倍超分输出1024x1024的Real-ESRGAN AnimeVideo-v3实现。AnimeVideo-v3使用残缺模型速度更快。生成效果和运算代价请自行斟酌。


## Onnx models: 
Please download converted models from the following link. You should be noticed that these model are originally developed in THA3， RIFE， waifu2x, and Real-SERGAN project. You can find licenses come with the model from the download.

请在下方链接中下载所有静态化模型并注意使用协议，放在项目的`/data` 文件夹中。

Please download and extract to `/data` folder for algorithm to run.

[Click here for download!](https://github.com/zpeng11/ezvtuber-rt/releases/download/0.0.1/20241220.zip)

## INT8 Quantization Research
By using PTQ(Post training quantization) methods, I calibrated and quantized THA3 model with ONNX Runtime quantization and Nvidia Model Optimizer. I worked on seperable fp16 models, and found that combiner and decomposer are good with quantization into INT8 with negligible error. However these two models are not frequently called in inference stage and so does not accelerate our Ezvtb core. Among morpher, rotator, and editor, I found that INT8 quantization causes huge error probably becauses of attention machanism. The only exception is that editor could work with qdq quantize mode only on Conv Layers of downsampling and upsampling stages, which brings partial quantization to ~20 Conv layers and introduced visible decline in generation quality. By comparing performance on Nvidia Nsight Visual Profiler, I found that partial quantization of editor did not bring noticable acceleration to my RTX3060 GPU in overall execution. 

Concludsion in brief: THA3 weights are not good for INT8 quantization in PTQ mothod, maybe try to do QAT method with using more dataset? 
## TODO

### INT8 Quantization and mobile deployment with NCNN.
## References
```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
@misc{Khungurn:2022,
    author = {Pramook Khungurn},
    title = {Talking Head(?) Anime from a Single Image 3: Now the Body Too},
    howpublished = {\url{http://pkhungurn.github.io/talking-head-anime-3/}},
    year = 2022,
    note = {Accessed: YYYY-MM-DD},
}

@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```
