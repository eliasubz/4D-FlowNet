from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SymmetricConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = F.pad(x, [self.pad] * 6, mode="reflect")
        return self.conv(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SymmetricConv3d(channels, channels, kernel_size=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SymmetricConv3d(channels, channels, kernel_size=3, bias=False),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PixelShuffle3d(nn.Module):
    """Rearrange channels into spatial dimensions for learned 3D upsampling.

    Transforms a tensor from ``(B, C·r³, D, H, W)`` to ``(B, C, D·r, H·r, W·r)``
    where *r* is the upscale factor.  This is the 3D extension of the sub-pixel
    convolution approach from Shi et al. (2016) used in the original 4DFlowNet
    paper for learned upsampling.
    """

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        r = self.r
        oc = c // (r ** 3)
        x = x.view(b, oc, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, oc, d * r, h * r, w * r)


class FourDFlowNet(nn.Module):
    """4DFlowNet 3D residual network.

    The paper used separate anatomical and velocity paths, LR residual blocks
    before upsampling, HR residual blocks after upsampling and three output
    heads for the velocity components.
    """

    def __init__(
        self,
        velocity_channels: int = 3,
        magnitude_channels: int = 3,
        width: int = 32,
        lr_blocks: int = 8,
        hr_blocks: int = 4,
        upsample_mode: str = "subpixel",
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode

        self.velocity_head = nn.Sequential(
            SymmetricConv3d(velocity_channels, width, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.anatomy_head = nn.Sequential(
            SymmetricConv3d(magnitude_channels, width, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.lr_fusion = nn.Sequential(
            SymmetricConv3d(width * 2, width, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.lr_body = nn.Sequential(*[ResidualBlock3D(width) for _ in range(lr_blocks)])

        if upsample_mode == "subpixel":
            # Learned upsampling: conv to expand channels, then rearrange to spatial.
            # width -> width * 2^3 channels, then PixelShuffle3d(2) -> width channels at 2x resolution.
            self.upsample = nn.Sequential(
                SymmetricConv3d(width, width * 8, kernel_size=3),
                PixelShuffle3d(upscale_factor=2),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsample = None

        self.hr_body = nn.Sequential(*[ResidualBlock3D(width) for _ in range(hr_blocks)])
        self.hr_refine = nn.Sequential(
            SymmetricConv3d(width, width, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.vx_head = nn.Sequential(SymmetricConv3d(width, 1, kernel_size=3), nn.Tanh())
        self.vy_head = nn.Sequential(SymmetricConv3d(width, 1, kernel_size=3), nn.Tanh())
        self.vz_head = nn.Sequential(SymmetricConv3d(width, 1, kernel_size=3), nn.Tanh())

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        velocity = lr[:, :3]
        magnitude = lr[:, 3:6]

        velocity_features = self.velocity_head(velocity)
        anatomy_features = self.anatomy_head(magnitude)
        features = self.lr_fusion(torch.cat([velocity_features, anatomy_features], dim=1))
        features = self.lr_body(features)

        if self.upsample is not None:
            features = self.upsample(features)
        else:
            features = F.interpolate(features, scale_factor=2, mode="trilinear", align_corners=True)

        features = self.hr_body(features)
        features = self.hr_refine(features)
        return torch.cat([self.vx_head(features), self.vy_head(features), self.vz_head(features)], dim=1)
