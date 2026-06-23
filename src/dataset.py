from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .mri_simulation import simulate_mri_acquisition
from .synthetic_flows import random_flow_field


class SyntheticFlowDataset(Dataset):
    """Synthetic LR input patches and HR velocity targets.

    The LR input has six channels: Vx, Vy, Vz, Mx, My, Mz. This follows the
    original paper's stated low-level inputs more closely than velocity only.

    When ``use_kspace_noise`` is True, the low-resolution input is generated
    using the paper's k-space simulation pipeline (FFT-based downsampling with
    complex Gaussian noise).  When False, the original trilinear downsample
    with additive Gaussian noise is used (faster, useful for quick demos).
    """

    def __init__(
        self,
        samples: int,
        hr_size: int = 32,
        scale: int = 2,
        noise_std: float = 0.03,
        seed: int = 0,
        use_kspace_noise: bool = False,
        snr_range: tuple[float, float] = (14.0, 17.0),
        venc_range: tuple[float, float] = (1.1, 3.0),
    ) -> None:
        self.samples = samples
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        self.noise_std = noise_std
        self.seed = seed
        self.use_kspace_noise = use_kspace_noise
        self.snr_range = snr_range
        self.venc_range = venc_range

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed + index)
        hr = random_flow_field(self.hr_size)
        torch.random.set_rng_state(generator_state)

        if self.use_kspace_noise:
            lr_input, hr = simulate_mri_acquisition(
                hr,
                scale=self.scale,
                snr_range=self.snr_range,
                venc_range=self.venc_range,
            )
        else:
            lr = F.interpolate(
                hr.unsqueeze(0),
                size=(self.lr_size, self.lr_size, self.lr_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

            if self.noise_std > 0:
                lr = lr + self.noise_std * torch.randn_like(lr)

            lr_velocity = lr.clamp(-1.0, 1.0)
            lr_magnitude = make_magnitude_channels(lr_velocity)
            lr_input = torch.cat([lr_velocity, lr_magnitude], dim=0)

        return lr_input.float(), hr.float()


def upsample_lr(lr: torch.Tensor, size: int, mode: str = "trilinear") -> torch.Tensor:
    # Extract velocity channels (first 3 channels)
    velocity = lr[:, :3] if lr.shape[1] > 3 else lr
    
    if mode == "trilinear":
        return F.interpolate(velocity, size=(size, size, size), mode="trilinear", align_corners=True)
        
    elif mode == "tricubic":
        import scipy.ndimage
        import numpy as np
        
        is_batched = (velocity.ndim == 5)
        if not is_batched:
            velocity = velocity.unsqueeze(0)
            
        b, c, d, h, w = velocity.shape
        device = velocity.device
        
        vel_np = velocity.detach().cpu().numpy()
        zoomed_batches = []
        zoom_factors = (size / d, size / h, size / w)
        
        for batch_idx in range(b):
            channels = []
            for ch_idx in range(c):
                upsampled = scipy.ndimage.zoom(vel_np[batch_idx, ch_idx], zoom=zoom_factors, order=3)
                channels.append(upsampled)
            zoomed_batches.append(np.stack(channels, axis=0))
            
        out_np = np.stack(zoomed_batches, axis=0)
        out_tensor = torch.from_numpy(out_np).to(device)
        return out_tensor if is_batched else out_tensor.squeeze(0)
        
    elif mode == "sinc":
        is_batched = (velocity.ndim == 5)
        if not is_batched:
            velocity = velocity.unsqueeze(0)
            
        b, c, d, h, w = velocity.shape
        device = velocity.device
        
        # 3D FFT (centered)
        fft_val = torch.fft.fftshift(torch.fft.fftn(velocity, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        
        # Zero padding
        padded_fft = torch.zeros((b, c, size, size, size), dtype=fft_val.dtype, device=device)
        start_d = (size - d) // 2
        start_h = (size - h) // 2
        start_w = (size - w) // 2
        padded_fft[:, :, start_d:start_d+d, start_h:start_h+h, start_w:start_w+w] = fft_val
        
        # 3D IFFT
        padded_fft_unshifted = torch.fft.ifftshift(padded_fft, dim=(-3, -2, -1))
        hr_reconstructed = torch.fft.ifftn(padded_fft_unshifted, dim=(-3, -2, -1)).real
        hr_reconstructed = hr_reconstructed * (size**3) / (d * h * w)
        
        return hr_reconstructed if is_batched else hr_reconstructed.squeeze(0)
        
    else:
        raise ValueError(f"Unknown upsampling mode: {mode}")



def make_magnitude_channels(velocity: torch.Tensor) -> torch.Tensor:
    """Synthetic magnitude channels from velocity (used in spatial-noise mode only)."""
    speed = torch.linalg.vector_norm(velocity, dim=0, keepdim=True).clamp(0.0, 1.0)
    vessel = torch.sigmoid(18.0 * (speed - 0.03))
    mx = vessel
    my = (0.85 * vessel + 0.15 * speed).clamp(0.0, 1.0)
    mz = (0.65 * vessel + 0.35 * speed).clamp(0.0, 1.0)
    return torch.cat([mx, my, mz], dim=0)
