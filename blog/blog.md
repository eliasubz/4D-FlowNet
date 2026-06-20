## 4DFlowNet: Paper Breakdown

This post will 1. Analyze the approach from 2020 and 2. Recreate the paper and results in a minimal setup. 

The simple goal of this paper is to improve the MRI imaging resolution of the bloodflow, especially for patients with abnormal flows, by increasing imaging time. 
Because datasets from imaging from real patients and they created high-quality simulations of the **thoracic aorta** using computational fluid dynamics (CFD) and sampled low-grade 16x16x16 patches. Fun Fact: They use blender to modify the 3D models and create more datasamples. The output of the 4DFlowNet is then a 2x upsampled (32x32x32) velocity field of the bloodflows, which achieved a relative errorr of 0.6-3.8%. 


This paper was not the first to adopt image super resolution using deep learning methods in the medical domain, but it was the first paper to work with velocity fields and 4D flow MRI representations. 

**The Network architecture** is based on a the generator of the super-resolution residual network (ResNet) paper [[2]](https://arxiv.org/abs/1609.04802). 

![4DFlowNet architecture](assets/4dflownet_architecture.webp)
