# Adaptive-Texture-Filtering-for-Single-Domain-Generalized-Segmentation
This repository provides the official PyTorch implementation of the following paper:  
**Adaptive Texture Filtering for Single Domain Generalized Segmentation**

## Congifuration Environment
- python 3.7
- pytorch 1.8.2
- torchvision 0.9.2
- cuda 11.1

## Before start
Download the GTA5, SYNTHIA, Cityscapes, BDD-100K, Mapillary datasets.

## Training and test
Run like this: python main.py --gta5_data_path /dataset/GTA5 --city_data_path /dataset/cityscapes --cuda_device_id 0  
Notice:   
-First training phase: load the model from model_firststage.py  
-Second training phase: load the model from model.py  
