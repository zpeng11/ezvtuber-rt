# eztuber-rt
> 不用3080，也不用80%占用，1660s大矿卡也能开着模型玩黑猴
> 
> Inspired and motivated by [THA3](https://github.com/pkhungurn/talking-head-anime-3-demo), [EeasyVtuber](https://github.com/yuyuyzl/EasyVtuber) and [RIFE](https://github.com/hzwer/ECCV2022-RIFE). Rapid implementation powered by Nvidia [TensorRT](https://github.com/NVIDIA/TensorRT) inference framework.
## Minimum Requeirement
### System
Windows only, Linux may work if you can figure out environment.

仅Windows
### Graphic Card
Nvidia Turing microarchitecture graphic cards only. Any gaming card higher than GeForce 16/20 series. For example all my tests are done on a GTX1660SUPER without any issue. Better gpu would give higher framerate, allow using higher percision model, and lower resource consumption.

仅支持16和20系以上的英伟达显卡，更好的显卡可以输出更高原生帧率，使用精度更高的模型和更少的系统占用。
## Key Improvements

### Static Models
Made static ONNX format model from THA3 and RIFE's prior dynamic PyTorch implementations, which allows graph level optimization and solid inference framework support. Removed dependency from PyTorch.

深度拆解 THA3 和 RIFE 模型并完成ONNX静态化。允许图优化及推理框架使用。去除 Pytorch 拥有更简单环境依赖。

### TensrRT
Powered by Nvidia TensorRT which unlocks the full computation potential of the GPU and acelerates using FP16, BF16, and TF32.

使用TenosorRT框架进行推理，可使用多种精度进行加速。

### RIFE
Introduced RIFE model to perform frame interpolation between 2 native frame comes out from THA. Resource consumption of RIFE is significantly less than THA. Just to noticed frame interpolation would bring system delay as a tradeoff. Provided up x4 times interpolation. This feature is crucial for low-end gpus like my 1660s.

使用RIFE模型进行实时插帧，带来更高帧数更少资源消耗，作为交换引入了一定系统延迟，请自行把控。低端卡福音！

## Onnx models: 
This project made static ONNX for models in THA3 and RIFE-lite-v4.25, please download converted models from the following link. You should be noticed that these model are originally developed in THA3 and RIFE project. You can find licenses come with the model from the download.

请在下方链接中下载所有静态化模型并注意使用协议，放在项目的`/data` 文件夹中。

Please download and extract to `/data` folder for algorithm to run.

[Click here for download!](https://drive.google.com/drive/folders/1cYj18EfVQ2Cl348_rkCu_fgaasHTI_io?usp=drive_link)


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
