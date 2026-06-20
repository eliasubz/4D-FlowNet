from __future__ import annotations

import torch


def endpoint_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target, dim=1).mean()


def peak_velocity_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_peak = torch.linalg.vector_norm(pred, dim=1).amax(dim=(1, 2, 3))
    target_peak = torch.linalg.vector_norm(target, dim=1).amax(dim=(1, 2, 3))
    return ((pred_peak - target_peak).abs() / target_peak.clamp_min(1e-6)).mean()


def flow_rate_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mid = pred.shape[-1] // 2
    pred_flow = pred[:, 2, :, :, mid].sum(dim=(1, 2))
    target_flow = target[:, 2, :, :, mid].sum(dim=(1, 2))
    return ((pred_flow - target_flow).abs() / target_flow.abs().clamp_min(1e-6)).mean()


def divergence(v: torch.Tensor) -> torch.Tensor:
    # v shape: [B, 3, D, H, W], using vx, vy, vz.
    dvx_dx = (v[:, 0, 1:-1, 1:-1, 2:] - v[:, 0, 1:-1, 1:-1, :-2]) * 0.5
    dvy_dy = (v[:, 1, 1:-1, 2:, 1:-1] - v[:, 1, 1:-1, :-2, 1:-1]) * 0.5
    dvz_dz = (v[:, 2, 2:, 1:-1, 1:-1] - v[:, 2, :-2, 1:-1, 1:-1]) * 0.5
    return dvx_dx + dvy_dy + dvz_dz


def divergence_l1(v: torch.Tensor) -> torch.Tensor:
    return divergence(v).abs().mean()
