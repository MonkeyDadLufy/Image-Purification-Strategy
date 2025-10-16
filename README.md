
<div align="center">
  
# A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy
</div>
The strategy for correcting the misaligned data pairs in the real-world ultra-low-dose CT dataset, along with a frequency-domain flow-matching model, and two sets of evaluation metrics.

The official implementation of paper [A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy](https://arxiv.org/pdf/2510.07492).

## Environment
create a new conda env,
and run
```
$ pip install -r requirements.txt
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

## Data Preprocessing

Modify the "input_directory" in Image Purification.py to be the parent directory of the gt and lq files in the dataset, 
and set "output_directory" to the output directory of the dataset after the IP strategy correction.
```
$ python Image Purification.py --input_directory [path1] --output_directory [path2]
```


## FFM Model 

### Training
Modify the path where the dataset is placed in FFM.yaml.

```
$ cd FFM
$ python main.py --mode train --config configs/FFM.yaml
```

Detailed training instructions will be updated soon.

### Sampling
Modify the model storage path and output image path in FFM.yaml

```
$ python main.py --mode test --config configs/FFM.yaml
```

### Model Evaluation
Modify the "calculate_result_all.py" file, changing "folder1" and "folder2" to represent the reference image folder and the image folder to be evaluated respectively.

```
$ cd..
$ python calculate_result_all.py
```


## Acknowledgement

This project is based on [Flow Matching](https://github.com/facebookresearch/flow_matching) and [MDMS](https://github.com/Oliiveralien/MDMS). Thanks for their awesome works.

### Contact
If you have any questions, please feel free to contact me via `onekey029@gmail.com`.
