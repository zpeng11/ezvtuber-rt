# eztuber-rt
> 不用3080，也不用80%占用，1660s大矿卡也能开着模型玩黑猴
> 
> Inspired and motivated by [THA3](https://github.com/pkhungurn/talking-head-anime-3-demo), [EeasyVtuber](https://github.com/yuyuyzl/EasyVtuber) and [RIFE](https://github.com/hzwer/ECCV2022-RIFE). Rapid implementation powered by Nvidia [TensorRT](https://github.com/NVIDIA/TensorRT) inference framework.
## Minimum Requeirement
### System
Windows only currently, *nix may work if you can figure out environment.

仅Windows
### Graphic Card
Most modern discrete gaming level graphic card! AMD RX580 and up, Intel Arc a750 and up, Nvidia GTX1650 and up. (Minimum standard is meet by reaching 30fps debug output after interpolation)

大部分现代游戏级独显均可在插帧下跑到30fps满足最小启动需求。如AMD RX580, 因特尔ARC A750, 英伟达 GTX1650。
## Key Improvements

### GPU Interface
Supporting DirectML for AMD and Intel GPU, cuda/TensorRT accelerate for Nvidia. DML implementation does not have good support on memory management due to ORT Python limitation.TensorRT supports Nvidia Turing microarchitecture graphic cards and above. Any gaming card higher than GeForce 16/20 series can get benefit from TensorRT and have advance memory management through PyCuda. For example all my tests are done on a GTX1660SUPER without any issue. Better gpu would give higher framerate, allow using higher percision model, and lower resource consumption.

A卡I卡使用DirectML, 有性能但不多，而且python的ort接口无法避免跨设备拷贝有性能损失。n卡可使用pycuda和TensorRT。TensorRT加速支持16和20系以上的英伟达显卡，配合cuda接口性能有跨越式提升(非严谨测试相同模型相比cuDNN实现提升约30%)，更好的显卡可以输出更高原生帧率，使用精度更高的模型和更少的系统占用。

### Static Models
Made static ONNX format model from THA3 and RIFE's prior dynamic PyTorch implementations, which allows graph level optimization and solid inference framework support. Removed dependency from PyTorch. Feel free to take these onnx models for your own inference framework. All models are generated from my [playground](https://github.com/zpeng11/talking-head-anime-3-demo/tree/start_testing). You can diff with main to see my work on it.

深度拆解 THA3 和 RIFE 模型并完成ONNX静态化。允许图优化及推理框架使用。去除 Pytorch 拥有更简单环境依赖和更快启动时间。如感兴趣请参考 [playground](https://github.com/zpeng11/talking-head-anime-3-demo/tree/start_testing) 了解静态化流程。

### TensrRT
Powered by Nvidia TensorRT which unlocks the full computation potential of the GPU and acelerates using FP16, BF16, and TF32.

使用TenosorRT框架进行推理，可使用多种精度进行加速。

### DirectML
Currently supporting AMD and Intel GPU by DirectML execution provider of OnnxRuntime. Due to Python API limitation, this method is not well optimized.

A卡和I卡使用OnnxRuntime 提供的DirectML支持，可用但因为Python接口不完善有诸多限制，此实现并非本项目主要实现方向，仅提供入门支持，请自行斟酌。

### Cache
Updated cache structure, provide VRAM+RAM+DiskDatabase solutions for caching results effectively lower down GPU resource comsumption. Use SIMD library TurboJPEG to save space.

实现显存，内存，和硬盘数据库等各式缓存方式供选择，有效减少gpu计算和显存占用。使用SIMD支持的TurboJPEG库实现快速图像压缩解压减少储存压力。


### RIFE
Introduced RIFE model to perform frame interpolation between 2 native frame comes out from THA. Resource consumption of RIFE is significantly less than THA. Just to noticed frame interpolation would bring system delay as a tradeoff. Provided up x4 times interpolation. This feature is crucial for low-end gpus like my 1660s. Limited by RIFE, interpolation up to x4 times, which brings 10fps native frame gen to 40fps experience.

使用RIFE模型进行实时插帧，带来更高帧数更少资源消耗，作为交换引入了一定系统延迟，请自行把控。低端卡福音！受制于RIFE模型设计，最高4倍插帧，等同于将10帧原生画面拉进40帧可用级别，即使高端卡也受益获得更低显卡占用，无需双卡双机。



## Onnx models: 
This project made static ONNX for models in THA3 and RIFE-lite-v4.25, please download converted models from the following link. You should be noticed that these model are originally developed in THA3 and RIFE project. You can find licenses come with the model from the download.

请在下方链接中下载所有静态化模型并注意使用协议，放在项目的`/data` 文件夹中。

Please download and extract to `/data` folder for algorithm to run.

[Click here for download!](https://drive.google.com/drive/folders/1cYj18EfVQ2Cl348_rkCu_fgaasHTI_io?usp=drive_link)

## TODO
### Fix FP16 with Onnxruntime DirectML
FP16 kernels has undetected bug on devices with FP16 support.
### Super Resolution for output
Multiple Anime Super Resolution models are under tests right now to find out efficiency and quality balance. This feature should be up soon :)
### INT8 Quantization and mobile deployment
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
```
