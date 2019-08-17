# **Shufflenetv2+ quantization(offline) model v1.0**

本网络采用shufflenetv2的结构，与dpc结构结合，进行图像语义分割的预测。并且将每一层进行bn融合，并对输入与权重、偏置做8比特的浮点量化。

### 准备

1.python>=2.7 for 64 bits windows or Linux

2.tensorflow-gpu==1.12.0

3.python一些第三方库:  numpy, opencv-python

4.一块gpu

### 文件

```
|--core
|  |--config.py 		//存储一些参数，里面cfg.QUANT为是否使用浮点量化的默认值。
|  |--nn_skeleton.py  	//构成模型的基础模块。含有bn的卷积有两个，bn融合有两个，都是一个是对于深度卷积的，一个是普通卷积的，与深度卷积相对应的都写了一个seperable。check是用来检测某一层的，某一层如果使用了check则会采用浮点量化，并保存一些信息。
|  |--save_parameters.py //保存每一层参数，暂时用不到
|  |--shuffle_net_v2.py //构成网络的几个结构。shuffle结构，dpc解码器结构，不用dpc的普通解码器（naive）机构，最终的预测卷积单元。如果对于某一层调试在这个文件中进行，可以把某一层的check设置为True。
|  |--model.py          //运行网络的文件入口。可以选择使用dpc结构或者不使用dpc结构。
|  |--test.png			//各种图片用来检验效果，test5是默认的。

```

### 运行

1.配置好tensorflow与第三方库

2.指定gpu运行程序model.py

### 网络结构

下图为含有dpc的网络结构

![幻灯片1](C:\Users\dell\Desktop\网络结构\幻灯片1.PNG)

不含有dpc结构的

![幻灯片2](C:\Users\dell\Desktop\网络结构\幻灯片2.PNG)

### 问题与调试

1.使用dpc结构，第一层卷积"Conv1"的融合后的weight范围在-17与21之间。当采取浮点量化时，如果采用clip为3的方法，即以16为界限，将绝对值大于16的4个数变成16，对结果有不小的影响。若选择clip为2，即以32为界限，效果也不好。

2.使用dpc结构，shufflenet的第一个stage的第一个下采样单元受浮点量化影响也比较严重。这个单元中有五个卷积，主要影响来自于其中的两个深度下采样卷积。

3.使用check调节程序：先进入shuffle_net_v2.py文件中进行设置，



