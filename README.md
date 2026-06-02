# <p align="center">DCP-YOLO: A Lightweight Anti-UAV Detection Algorithm Based on YOLOv26n</p>

 <p align="center">Jia Liu, Die Mao, Zirui Jiang, Yihan Cao, and Dapeng Chen</p>
  <p align="center">Nanjing University of Information Science & Technology</p>

## <p align="center">ABSTRACT</p>
With the rapid development of the low-altitude economy, illegal Unmanned Aerial Vehicle (UAV) flights have brought serious security risks. Existing anti-UAV detection methods still face several challenges, including the loss of small target features, limited computational resources on edge devices, and difficulty in balancing detection accuracy with model lightweighting. To address these problems, this paper proposes a lightweight multi-module collaborative optimization detection method based on YOLOv26n. First, a lightweight feature extraction structure named DPC3k2 is designed to enhance the multi-directional feature extraction capability for small targets while reducing the number of parameters. Then, a learnable upsampling module, ConverseSample, is proposed to recover high-frequency texture details of targets. Finally, a C2PSCAA attention-enhanced module is constructed to improve the multi-scale feature learning capability of the model. Experimental results on multiple public datasets show that the proposed method contains only 2.29M parameters and achieves an inference speed of 206.5 Frames Per Second (FPS), while significantly improving detection accuracy compared with the baseline model. Ablation and generalization experiments further verify the effectiveness and robustness of each module. The proposed method effectively balances detection accuracy and lightweight performance, making it suitable for real-time low-altitude UAV detection tasks in resource-constrained scenarios.

## <p align="center"> FRAMEWORK OF DCP-YOLO</p>
The diagram is shown below， this paper proposes three lightweight improvements: the DPC3k2 module combines depth-separable convolutions with windmill convolutions to enhance multi-directional feature extraction for small drone targets whilst reducing the number of parameters; the ConverseSample learnable upsampling module transforms traditional interpolation into an optimisation problem, effectively restoring edge textures and avoiding checkerboard artefacts; the C2PSCAA module introduces context-anchored attention to enhance the network’s spatial-directional and multi-scale semantic learning capabilities, thereby improving detection robustness in complex backgrounds.

![image](https://github.com/md123-p/yolo/blob/main/fig2.jpg)
## <p align="center">DETAILS OF IMPLEMENT</p>
This project includes scripts for requirements, model training and testing.

### Requirements

- Python 3.8.20
- torch 2.4.1
- CUDA 12.1

### Dataset
We use three public datasets, including DUT Anti-UAV, TIB-Net, and VisDrone2019 
For more details, see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9785379: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9141228; https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9573394

### Model Training
Use `train.py` to train the model. Modify the dataset path in `train.py`. Adjust parameters such as epoch and batch_size based on the training environment.
```bash
python train.py
```
After training, the model will be saved in the specified directory.

## Testing
Use `get_fps.py` to get FPS.
```bash
python get_fps.py
```

