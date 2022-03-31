# Ship Detection based on YOLOv3 and KV260

## Overview

This is the entry project of the Xilinx Adaptive Computing Challenge 2021. It uses YOLOv3 for ship target detection in optical remote sensing images, and deploys DPU on the KV260 platform to achieve hardware acceleration. 

First, I trained the YOLOv3 model for ship detection using the darknet framework. Secondly, referring to the darknet_yolov4 tutorial of Xilinx, I quantified and compiled the network model and evaluated the quantized model. Finally, I deployed the compiled xmodel to the KV260 platform with PYNQ framework to implement the hardware-accelerated YOLOv3 network for ship detection tasks. The model with quantized and compiled performs well on KV260 and achieves 7.63 FPS. In the follow-up work, I will apply methods such as compression pruning to the model to improve the running speed of the model.

## Things used in this project

### Hardware components

* AMD-Xilinx Kria KV260 Vision AI Starter Kit
* A computer with a high-performance GPU such as an RTX3090 to accelerate model training

### Software apps and online services

* AMD-Xilinx PYNQ Framework
* AMD-Xilinx Xilinx Vitis-AI

## References

* [Vitis AI Github Repository](https://github.com/Xilinx/Vitis-AI).
* [Vitis AI Tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials).
* [Kria PYNQ Github Repository](https://github.com/Xilinx/Kria-PYNQ).
* [AlexeyAB Darknet](https://github.com/AlexeyAB/darknet).

## How to reproduce this project

Follow the steps below to reproduce it.

<img src="C:\Users\lyk\AppData\Roaming\Typora\typora-user-images\image-20220331222609540.png" alt="image-20220331222609540" style="zoom: 15%;" />

### 1. Prepare KV260 and PYNQ

Refer to [Kria PYNQ Github Repository](https://github.com/Xilinx/Kria-PYNQ).

#### Get the Ubuntu SD Card Image

Follow the steps to [Get Started with Kria KV260 Vision AI Starter Kit](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit/kv260-getting-started-ubuntu/setting-up-the-sd-card-image.html) until you complete the [Booting your Starter Kit](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit/kv260-getting-started-ubuntu/booting-your-starter-kit.html) section.

#### Install PYNQ

Then install PYNQ on your Kria KV260 Vision AI Starter Kit. Simply clone this repository from the KV260 and run the install.sh script.

```bash
git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ/
sudo bash install.sh
```

This script will install the required Debian packages, create Python virtual environment and configure a Jupyter portal. This process takes around 25 minutes.

#### Open Jupyter

JupyterLab can now be accessed via a web browser `<ip_address>:9090/lab` or `kria:9090/lab`. The password is **xilinx**.

### 2. Run Notebook with PYNQ

Clone this Repository on Kria-PYNQ directory and run `yolov3_dpu/dpu_yolov3_voc.ipynb`.

Image test:

![output_22_1](D:\lyk\Downloads\dpu_yolov3_voc\output_22_1.png)

## Train & Deploy YOLOv3 on KV260 with PYNQ

Clone [AlexeyAB Darknet](https://github.com/AlexeyAB/darknet), and train your own YOLOv3 model.

<img src="C:\Users\lyk\AppData\Roaming\Typora\typora-user-images\image-20220331215455126.png" alt="image-20220331215455126" style="zoom: 67%;" />

For ship detection, my model have 93.1% mAP  with 0.5 IoU thresh.

Then launch Vitis-AI-CPU-1.4 docker environment, run scripts in folder to quantize and compile xmodel.

```bash
conda activate vitis-ai-caffe
bash scripts/darknet_convert.sh # convert darknet model to caffe model
```

![image-20220331221545009](C:\Users\lyk\AppData\Roaming\Typora\typora-user-images\image-20220331221545009.png)

```bash
bash scripts/run_vai_q.sh # quantize model
```

<img src="C:\Users\lyk\AppData\Roaming\Typora\typora-user-images\image-20220331222002206.png" alt="image-20220331222002206" style="zoom:67%;" />

```bash
bash run_vai_c_kv260.sh # compile caffe model to xmodel
```

<img src="C:\Users\lyk\AppData\Roaming\Typora\typora-user-images\image-20220331222155197.png" alt="image-20220331222155197" style="zoom:67%;" />



