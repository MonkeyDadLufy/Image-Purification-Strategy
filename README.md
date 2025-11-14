
<div align="center">
  
# A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy
</div>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![NumPy](https://img.shields.io/badge/NumPy-2.3-blue?logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-3-44A833?logo=anaconda&logoColor=white)](https://www.anaconda.com/)

This repository contains the official implementation of our paper [A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy](https://arxiv.org/pdf/2510.07492).

<div align="center">
  <img src="assets/motivation.png" alt="Motivation Overview" width="800">
  <br>
</div>

## Overview

Our framework introduces:
- An innovative Image Purification (IP) strategy for correcting misaligned data pairs in real-world ultra-low-dose CT datasets
- A frequency-domain flow-matching model for superior denoising performance
- Comprehensive evaluation metrics for result assessment

## Installation

1. Create and activate a new conda environment:
```bash
conda create -n ffm python=3.8
conda activate ffm
```

2. Install required packages:
```bash
pip install -r requirements.txt
```


The dataset folders are structured in the following way:
```
.
├── dataset                 
│   ├── train                      
│   │   ├── gt
│   │   └── lq                     
│   ├── val 
│   │   ├── gt
│   │   └── lq
└── └── test                     
        ├── gt
        └── lq

```

## 🧹 Data Preprocessing

<div align="center">
  <img src="assets/Image_Purification.png" alt="Image Purification (IP) Strategy" width="800">
  <br>
  <i>Image Purification (IP) workflow</i>
</div>

path1 is the parent directory of the gt and lq files in the dataset, 
and path2 is the output directory of the dataset after the IP strategy correction.

```
$ python Image Purification.py --input_directory [path1] --output_directory [path2]
```


## 🚀 FFM Model Training and Testing

### 🏋️ Training

1. Configure the dataset paths in `FFM/configs/FFM.yaml`
2. Start training:
```bash
cd FFM
python main.py --mode train --config configs/FFM.yaml
```

For a quick test run, use the smoke test configuration:
```bash
python main.py --mode train --config configs/FFM_smoke_3ep.yaml
```

### 🧪 Testing

1. Update the model path and output directory in `FFM.yaml`
2. Run inference:
```bash
python main.py --mode test --config configs/FFM.yaml
```

### 📊 Evaluation

To evaluate the model performance:

1. Edit `calculate_result_all.py`:
   - Set `folder1` as the path to ground truth images
   - Set `folder2` as the path to generated images
2. Run evaluation:
```bash
python calculate_result_all.py
```

The script will compute and display various metrics including FSIM, SSIM, and more.


## 🧩 Configuration Details

The model behavior can be customized through the following configuration files:
- `FFM/configs/FFM.yaml`: Main configuration for full training
- `FFM/configs/FFM_smoke_3ep.yaml`: Quick test configuration with 3 epochs

Key configuration parameters include:
- Learning rate and batch size
- Training epochs and validation frequency
- Data paths and model saving options
- ODE solver parameters for inference

## 🙏 Acknowledgements

This project builds upon the following excellent works:
- [Flow Matching](https://github.com/facebookresearch/flow_matching)
- [MDMS](https://github.com/Oliiveralien/MDMS)

## 📄 License

This project is released under the MIT License.

## 📬 Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/MonkeyDadLufy/Image-Purification-Strategy/issues)
- Contact via email: `onekey029@gmail.com`

## 📚 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{gong2025denoising,
  title={A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy},
  author={Gong, Guoliang and Yu, Man},
  journal={arXiv preprint arXiv:2510.07492},
  year={2025}
}
