## 4DFlowNet: Paper Breakdown

This post analyzes the 4DFlowNet (2020) paper and presents a small implementation of the paper. 

### Introduction

The goal of this paper is to improve the 4D flow MRI resolution. 4D Flow MRI measures both the velocity and direction over time of moving blood but is subject to noise and low resolution. 4D Flow MRI scans from real patients are expensive which moved the authors to sample realistic low-grade 16x16x16 MRI patches from a  high-quality simulation of the **thoracic aorta** using computational fluid dynamics (CFD). These simulated scanes were upscaled with an image super resolution network that was adapted to 4D flow MRI velocity field representations. The 4D FlowNet trained on simulated data and demonstrated flow rate measurements giving an error of 1.1–3.8% in real volunteer data.

Fun Fact: They use [blender](https://www.blender.org/) to modify the 3D models and create more datasamples. 

### Methodolgy

**The Network architecture** is based on a the generator of the super-resolution residual network (ResNet) paper [[2]](https://arxiv.org/abs/1609.04802). The input has one path for the raw noisy velocity maps (16^3) and one that accepts calculated context: Magnitude, Velocities and PC-MRA (=Mag*Speed). Both of them pass through two convolutional maps and are then concatenated, after which they are passed through 8 residual blocks (RB). Each RB consist of two conv layer with a nonlinear layer inbetween and a skip-connection after which another Leaky ReLU is used. After the 8 RB-layers we have 64 sematincally rich feature maps that are upsampled using a simple bilinear resize, after which we pass them through 4 more RBs and pass it through three two-layer convolutional heads that convolve the 64 channels back into 1 for the xyz-compontents of the velocities.


![4DFlowNet architecture](assets/4dflownet_architecture.webp)


**The loss function** consist of a simple mean-squared error (MSE) loss, compares the upsampled velocity components with their ground-truth, added with a velocity gradient (VG) term. While the MSE loss is used to vector maginude error it can lead to blurry images by missing high-frequency spatial velocity shifts, which leads to poor performance close to the vessel walls. The VG loss counter-acts that by rewarding accurate directional derivatives between adjacent velocity vectors. 
$$L_{total} = l_{MSE} + 10^{-3} \cdot l_{VG}$$

**Generating accurate 4D MRI images** needed much more than just down-sampling the mesh from a perfect CFD simulation. In order to model the *Rayleigh noise* that MRI images are subject to, the researchers added noise in the frequency domain using fast fourier transform. Specifically, they did that by calculating the complex numbers from the phase and magnitude images, converting the complex numbers into frequency domain (k-space) and truncating the high-frequency information along all three axes. In addition, they added some white noise to the frequencies and converted them back to the spatial domain.

**The Metrics** that were used in the paper are **relative speed error**, which compares the predicted speed with the ground truth, **net flow rate** (mL/s), which is calculate by integrating the velocity vectors passing through a cross section plane and the **divergence Field** which tracks how well the model preserves divergence-free nature of an incompressible fluid. 


### Paper Results

During the experiments there where three breathtaking results. The most predictable results was that on the **synthetic data** 4D FlowNet outperformed traditional mathematical interpolation methods (Linear, Cubic, Sinc). These were struggling especially during low-velocity phases, while 4D FlowNet was not. Finally, upsampling an In-Vivo aorta from a healthy volunteer cleanly isolated the tissue boundaries and was free from stitching artifacts. 

![Comparison of different upsampling methods](assets/4dfn_vs_math.png)


The visual breakdown above showcases how the 4D FlowNet outperforms simple upsampling baselines. While the math-based interpolation methods upsample the noise, the super-resolution generator learned how to effectively subtract the noise, improve the quality, and even implicitly model the fluid's divergence properties. 

---

### Improving 4DFlowNet with Sub-Pixel Convolutions (Our Upgrade)

While reproducing the paper, we noticed a key discussion point on Page 12: the authors actually *wanted* to use **PixelShuffle (sub-pixel convolution)** but rejected it. They wrote that in 3D, sub-pixel convolutions:
1. Caused severe checkerboard artifacts.
2. Suffered from poor training convergence.

Because of this, they settled on a fixed trilinear resize layer inside the neural network.

#### How We Solved It
We successfully implemented 3D Sub-Pixel Convolution (`PixelShuffle3d`) by introducing three modern deep learning techniques that didn't exist or weren't standard in 2020:
* **AdamW Weight Decay**: Helps stabilize weights in the expanding upsampling layers.
* **Cosine Annealing Learning Rate Schedule**: Smooths out late-stage convergence.
* **Gradient Clipping (`max_norm=1.0`)**: Capping gradients blocks the high-frequency backprop shocks that cause checkerboard artifacts.

#### The Results (K-Space Dataset)

When evaluated on the identical physics-based K-Space validation dataset, we can observe the difference between direct mathematical interpolation and learned deep super-resolution:

| Upsampling Method | Val MAE | Peak Velocity Err% | Net Flow Err% |
| :--- | :--- | :--- | :--- |
| **Trilinear (Math Baseline)** | ~0.226 | ~30.1% | ~50.1% |
| **Tricubic (Math Baseline)** | ~0.198 | ~28.5% | ~46.2% |
| **Sinc (Math Baseline)** | ~0.224 | ~29.8% | ~49.8% |
| **Trilinear Model (Paper)** | ~0.006 | ~7.4% | ~6.6% |
| **Sub-Pixel Model (Ours)** | **~0.005** | **~7.0%** | **~5.9%** |

*Note: Mathematical baselines struggle severely because they cannot unwrap phase aliasing or denoise Rayleigh magnitude distributions, whereas the deep networks excel. Our sub-pixel model reduces the remaining error by ~15% over the paper's trilinear architecture.*

#### Correct Visualization via Clinical Masking
In real MRI post-processing, background noise (air and stationary tissue) is masked using a magnitude threshold. Without masking, the background contains random velocity vectors, which makes it look like blood is flowing outside the vessel walls. 

By applying a binary fluid mask derived from the magnitude/ground-truth channel in our 3D visualizer, we can cleanly segment the aorta. This guarantees that all velocity cones remain strictly inside the translucent vessel walls. 