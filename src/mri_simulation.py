"""K-space MRI simulation following the 4DFlowNet paper.

The preprocessing pipeline converts a high-resolution velocity field into a
low-resolution, noisy magnitude and phase image pair by simulating the MRI
acquisition process:

    HR velocity
    -> phase encode with VENC: phase = v / venc * pi
    -> form complex signal: mag * exp(j * phase)
    -> FFT to k-space
    -> crop high frequencies (downsampling)
    -> add complex Gaussian noise (becomes Rayleigh noise in image domain)
    -> IFFT back to image domain
    -> extract magnitude and phase
    -> decode velocity: v = phase / pi * venc

Reference: Ferdian et al., "4DFlowNet: Super-Resolution 4D Flow MRI Using
Deep Learning and Computational Fluid Dynamics", Frontiers in Physics, 2020.
Original NumPy implementation:
    github.com/EdwardFerdian/4DFlowNet/blob/master/src/prepare_data/fft_downsampling.py
"""
from __future__ import annotations

import math

import torch


def rectangular_crop_3d(kspace: torch.Tensor, crop_ratio: float) -> torch.Tensor:
    """Center-crop a 3D k-space tensor to downsample by *crop_ratio*.

    Args:
        kspace: Complex tensor of shape ``(D, H, W)``.
        crop_ratio: Fraction of each axis to keep (e.g. 0.5 for 2× downsample).

    Returns:
        Cropped complex tensor.
    """
    d, h, w = kspace.shape
    # Shift zero-frequency to center, crop, shift back.
    shifted = torch.fft.fftshift(kspace)

    cd = int(d // 2 * crop_ratio)
    ch = int(h // 2 * crop_ratio)
    cw = int(w // 2 * crop_ratio)

    cropped = shifted[
        d // 2 - cd : d // 2 + cd,
        h // 2 - ch : h // 2 + ch,
        w // 2 - cw : w // 2 + cw,
    ]
    return torch.fft.fftshift(cropped)


def add_complex_kspace_noise(
    kspace: torch.Tensor, target_snr_db: float
) -> torch.Tensor:
    """Add complex Gaussian noise to k-space at a target SNR.

    Following Gudbjartsson et al. 1995, noise is added directly to the complex
    signal.  This produces Rayleigh-distributed noise in the magnitude image
    and uniform noise in the phase image in background regions — matching
    real MRI acquisition characteristics.

    Args:
        kspace: Complex tensor (any shape).
        target_snr_db: Target signal-to-noise ratio in decibels.

    Returns:
        Noisy complex tensor of the same shape.
    """
    signal_power = torch.mean(torch.abs(kspace) ** 2)
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    sigma = torch.sqrt(noise_power)

    noise = sigma * torch.randn(
        *kspace.shape, 2, device=kspace.device, dtype=kspace.real.dtype
    )
    noise_complex = torch.complex(noise[..., 0], noise[..., 1])
    return kspace + noise_complex


def _rescale_magnitude(new_mag: torch.Tensor, old_size: int) -> torch.Tensor:
    """Rescale magnitude after k-space cropping to preserve energy."""
    new_size = new_mag.numel()
    ratio = new_size / (old_size**3)
    return new_mag * ratio


def downsample_phase_image(
    velocity: torch.Tensor,
    magnitude: torch.Tensor,
    venc: float,
    crop_ratio: float,
    target_snr_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample a single velocity component through k-space simulation.

    This is the core pipeline matching the paper's ``downsample_phase_img``:
        velocity -> phase -> complex -> FFT -> crop -> noise -> IFFT -> decode

    Args:
        velocity: Real tensor of shape ``(D, H, W)`` — one velocity component.
        magnitude: Real tensor of shape ``(D, H, W)`` — magnitude image.
        venc: Velocity encoding value (m/s).  Phase = velocity / venc * π.
        crop_ratio: Fraction of k-space to keep per axis (0.5 = 2× downsample).
        target_snr_db: Target SNR in decibels.

    Returns:
        ``(lr_velocity, lr_magnitude)`` — both real tensors at the downsampled
        resolution.
    """
    original_size = velocity.shape[0]

    # Phase-encode velocity.
    phase = velocity / venc * math.pi

    # Form complex MRI signal: magnitude * exp(j * phase).
    complex_img = magnitude * torch.exp(1j * phase)

    # FFT to k-space.
    kspace = torch.fft.fftn(complex_img)

    # Crop high frequencies (downsample).
    kspace = rectangular_crop_3d(kspace, crop_ratio)

    # Add complex Gaussian noise in k-space.
    kspace = add_complex_kspace_noise(kspace, target_snr_db)

    # IFFT back to image domain.
    new_complex = torch.fft.ifftn(kspace)

    # Extract magnitude and phase.
    new_mag = torch.abs(new_complex)
    new_mag = _rescale_magnitude(new_mag, original_size)
    new_phase = torch.angle(new_complex)

    # Decode velocity from phase.
    new_velocity = new_phase / math.pi * venc

    return new_velocity.real.float(), new_mag.real.float()


def simulate_mri_acquisition(
    hr_velocity: torch.Tensor,
    scale: int = 2,
    snr_range: tuple[float, float] = (14.0, 17.0),
    venc_range: tuple[float, float] = (1.1, 3.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate the full MRI acquisition pipeline for a 3-component velocity field.

    Creates per-axis magnitude images from the vessel mask (speed > threshold),
    randomizes VENC and SNR per-sample, and runs the k-space downsampling for
    each velocity component independently (as in the original paper).

    Args:
        hr_velocity: Tensor ``[3, D, H, W]`` with vx, vy, vz in [-1, 1].
        scale: Spatial downsampling factor (default 2).
        snr_range: ``(min_snr_db, max_snr_db)`` — randomized per sample.
        venc_range: ``(min_venc, max_venc)`` — randomized per sample.

    Returns:
        ``(lr_input, hr_target)`` where ``lr_input`` has shape
        ``[6, D/s, H/s, W/s]`` (3 velocity + 3 magnitude channels) and
        ``hr_target`` is the original ``hr_velocity``.
    """
    crop_ratio = 1.0 / scale
    device = hr_velocity.device

    # Random SNR (uniform in dB, matching paper's randint(140,170)/10).
    target_snr_db = (
        torch.empty((), device=device).uniform_(snr_range[0], snr_range[1]).item()
    )

    # Random VENC — must be above max velocity to avoid aliasing.
    max_vel = hr_velocity.abs().amax().item()
    venc_min = max(venc_range[0], max_vel * 1.1)
    venc_max = max(venc_range[1], venc_min + 0.1)
    venc = torch.empty((), device=device).uniform_(venc_min, venc_max).item()

    # Create magnitude image from vessel mask (speed > threshold).
    speed = torch.linalg.vector_norm(hr_velocity, dim=0)
    vessel_mask = (speed > 0.02).float()

    # Vary magnitude level per sample (paper uses values from [60,80,120,180,240]).
    mag_level = torch.empty((), device=device).uniform_(60.0, 240.0).item()
    magnitude = vessel_mask * mag_level

    # Process each velocity component through the k-space pipeline.
    lr_velocities = []
    lr_magnitudes = []
    for c in range(3):
        lr_v, lr_m = downsample_phase_image(
            hr_velocity[c], magnitude, venc, crop_ratio, target_snr_db
        )
        lr_velocities.append(lr_v)
        lr_magnitudes.append(lr_m)

    lr_velocity = torch.stack(lr_velocities, dim=0)
    lr_magnitude = torch.stack(lr_magnitudes, dim=0)

    # Normalize velocity to [-1, 1] for network input.
    vel_max = lr_velocity.abs().amax().clamp_min(1e-6)
    lr_velocity = lr_velocity / vel_max

    # Normalize magnitude to [0, 1].
    mag_max = lr_magnitude.amax().clamp_min(1e-6)
    lr_magnitude = lr_magnitude / mag_max

    lr_input = torch.cat([lr_velocity, lr_magnitude], dim=0)
    return lr_input.float(), hr_velocity.float()
