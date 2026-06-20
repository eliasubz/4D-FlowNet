from __future__ import annotations

import torch
import torch.nn.functional as F


def velocity_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper-style velocity-gradient loss for Vx, Vy, Vz.

    This compares dVx/dx, dVy/dy, and dVz/dz using centered finite differences.
    """
    pred_dvx_dx = pred[:, 0, 1:-1, 1:-1, 2:] - pred[:, 0, 1:-1, 1:-1, :-2]
    true_dvx_dx = target[:, 0, 1:-1, 1:-1, 2:] - target[:, 0, 1:-1, 1:-1, :-2]

    pred_dvy_dy = pred[:, 1, 1:-1, 2:, 1:-1] - pred[:, 1, 1:-1, :-2, 1:-1]
    true_dvy_dy = target[:, 1, 1:-1, 2:, 1:-1] - target[:, 1, 1:-1, :-2, 1:-1]

    pred_dvz_dz = pred[:, 2, 2:, 1:-1, 1:-1] - pred[:, 2, :-2, 1:-1, 1:-1]
    true_dvz_dz = target[:, 2, 2:, 1:-1, 1:-1] - target[:, 2, :-2, 1:-1, 1:-1]

    return (
        F.mse_loss(pred_dvx_dx, true_dvx_dx)
        + F.mse_loss(pred_dvy_dy, true_dvy_dy)
        + F.mse_loss(pred_dvz_dz, true_dvz_dz)
    )


def four_d_flow_loss(pred: torch.Tensor, target: torch.Tensor, gradient_weight: float = 1e-3) -> torch.Tensor:
    return F.mse_loss(pred, target) + gradient_weight * velocity_gradient_loss(pred, target)
