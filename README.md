# L2CS-Net: Fine-Grained Gaze Estimation
Gaze Estimation

### Structure

- `nn.py`: Defines the L2CS neural network architecture.
- `util.py`: Contains utility functions and classes.
- `datasets.py`: Handles data loading, preprocessing, and augmentation.
- `main.py`: The main executable script that sets up the model, performs training,testing, and inference.

### Installation

```
conda create -n PyTorch python=3.9
conda activate PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python==4.5.5.64
pip install scipy
pip install tqdm
pip install timm
```

### Gaze360 Dataset Preparation

- `Download Gaze360 dataset from` [here](http://gaze360.csail.mit.edu/download.php).

- `Apply data preprocessing from` [here](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).

### Train

* Configure your dataset path in `main.py` for training
* Run `python main.py --train` for Single-GPU training
* Run `bash main.sh $ --train` for Multi-GPU training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Demo

* Configure your video path in `main.py` for visualizing the demo
* Run `python main.py --demo` for demo

### Results

| Backbone  | Epochs |  MAE |                                                                                Model |
|:---------:|:------:|-----:|-------------------------------------------------------------------------------------:|
| ResNet18  |  120   | 11.1 | [**weight**](https://github.com/Shohruh72/L2CSNet/releases/download/v.1.0.0/best.pt) |
| ResNet18* |  120   | 12.2 |                                                                                      |

`*` means that the results are from original repo, see reference

#### Reference

* https://github.com/Ahmednull/L2CS-Net
