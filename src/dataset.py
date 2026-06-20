from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .synthetic_flows import random_flow_field


class SyntheticFlowDataset(Dataset):
    """Synthetic LR input patches and HR velocity targets.

    The LR input has six channels: Vx, Vy, Vz, Mx, My, Mz. This follows the
    original paper's stated low-level inputs more closely than velocity only.
    """

    def __init__(
        self,
        samples: int,
        hr_size: int = 32,
        scale: int = 2,
        noise_std: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed + index)
        hr = random_flow_field(self.hr_size)
        torch.random.set_rng_state(generator_state)

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


def upsample_lr(lr: torch.Tensor, size: int) -> torch.Tensor:
    velocity = lr[:, :3] if lr.shape[1] > 3 else lr
    return F.interpolate(velocity, size=(size, size, size), mode="trilinear", align_corners=True)


def make_magnitude_channels(velocity: torch.Tensor) -> torch.Tensor:
    speed = torch.linalg.vector_norm(velocity, dim=0, keepdim=True).clamp(0.0, 1.0)
    vessel = torch.sigmoid(18.0 * (speed - 0.03))
    mx = vessel
    my = (0.85 * vessel + 0.15 * speed).clamp(0.0, 1.0)
    mz = (0.65 * vessel + 0.35 * speed).clamp(0.0, 1.0)
    return torch.cat([mx, my, mz], dim=0)
