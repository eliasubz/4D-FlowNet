# 4DFlowNet Mini

A clean, PyTorch-based reproduction and extension of the core **4DFlowNet** methodology on synthetic blood-flow data:

1. **Synthetic Blood-Flow Field Generation**: Creates 3D analytic velocity profiles representing blood flow.
2. **K-space MRI Simulation**: Simulates the physical MRI acquisition process (VENC phase encoding, FFT k-space truncation, complex Gaussian noise, and IFFT reconstruction).
3. **Deep Residual Network**: Super-resolves and denoises 4D flow velocity fields using a ResNet.
4. **Sub-Pixel Upsampling**: Implements learned 3D sub-pixel convolution (`PixelShuffle3d`) to replace fixed trilinear resizing inside the network.

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Upgraded Training CLI
Train the network on the physically realistic k-space MRI dataset using learned sub-pixel convolution:
```bash
python train.py --epochs 15 --train-samples 1024 --val-samples 128 --batch-size 4 --use-kspace-noise --upsample-mode subpixel
```
If a GPU is available, the script automatically enables mixed precision (`torch.amp`) and GPU acceleration.

To run the backward-compatible trilinear upsampling path (the baseline paper design):
```bash
python train.py --epochs 15 --train-samples 1024 --val-samples 128 --batch-size 4 --use-kspace-noise --upsample-mode trilinear
```

---

## Files

- `src/synthetic_flows.py` — Generates synthetic 3D flow velocities with randomized stenosis, swirl, and branch profiles.
- `src/mri_simulation.py` — Physics-based simulator mapping velocities to complex k-space, applying truncation and noise.
- `src/dataset.py` — Custom PyTorch dataset integrating the k-space simulation.
- `src/model.py` — ResNet model with dual velocity/anatomical paths and configurable upsampling modes (`subpixel` vs. `trilinear`).
- `src/losses.py` — Implements voxel-wise MSE and velocity-gradient losses.
- `src/metrics.py` — Evaluates MAE, Peak Velocity, and Net Flow Rate errors.
- `src/visualize.py` — Generates 3D translucent vessel walls and velocity vector quivers using Plotly.
- `train.py` — Training interface supporting AdamW, Cosine Annealing, and Gradient Clipping.
- `4dflownet_colab.ipynb` — The primary Google Colab entry point. Runs the **Trilinear (Paper) vs. Sub-Pixel (Upgraded)** ablation sweep and outputs the interactive 3D visualizations.
