# Learnable Pooling in Graph Convolution Networks for Brain Surface Analysis

This repository contains the pytorch implementation of our paper titled "Adaptive Graph Convolution Pooling for Brain Surface Analysis" published in IPMI 2019. "Learnable Pooling in Graph Convolution Networks for Brain Surface Analysis", is an extension of this work currently under review at IEEE-PAMI. This work proposes a new learnable graph pooling method for processing multiple surface-valued data to output subject-based information. The proposed method innovates by learning an intrinsic aggregation of graph nodes based on graph spectral embedding.

The [Graph Spectreal alignment](https://github.com/kharitz/spectral_alignment.git) repository reads the FreeSurfer processed brain mesh and generates the aligned spectral features, embeddings, and labels necessary for this work. 

### Where to find the dataset?
- The MindBoggle dataset is available to download [here](https://osf.io/nhtur/).
- The ADNI dataset is available to download [here](http://adni.loni.ucla.edu).

### Package Requirements
- PyTorch >1.4.0
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) >1.4.3 and dependencies
- requirements.txt
- [GraphSpectreal alignment](https://github.com/kharitz/spectral_alignment.git)

### Usage
1. Download the datasets and generate the brain graph data using [GraphSpectreal alignment](https://github.com/kharitz/spectral_alignment.git) repository.
2. Set paramenters and add the path to the generated data folder in the learnpool/config/config.json file
3. The learnable pooling model for disease classification/ age regression can be trained using:  
```
python3 main.py --config ./learnpool/config/config.json --gpu 0
```


#### REFERENCE 
Please cite our paper and PyTorch Geometric repository if you use this code in your own work:

```
@inproceedings{gopinath2019adaptive,
  title={Adaptive graph convolution pooling for brain surface analysis},
  author={Gopinath, Karthik and Desrosiers, Christian and Lombaert, Herve},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={86--98},
  year={2019},
  organization={Springer}
}
```
