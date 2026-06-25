from __future__ import annotations

import math

import torch


def make_grid(size: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return x, y, z


def _random_rotation(device: torch.device | str) -> torch.Tensor:
    ax = torch.empty((), device=device).uniform_(-0.65, 0.65)
    ay = torch.empty((), device=device).uniform_(-0.65, 0.65)
    az = torch.empty((), device=device).uniform_(-math.pi, math.pi)

    sx, cx = torch.sin(ax), torch.cos(ax)
    sy, cy = torch.sin(ay), torch.cos(ay)
    sz, cz = torch.sin(az), torch.cos(az)

    rx = torch.stack(
        [
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
            torch.stack([torch.zeros_like(cx), cx, -sx]),
            torch.stack([torch.zeros_like(cx), sx, cx]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, torch.zeros_like(cz)]),
            torch.stack([sz, cz, torch.zeros_like(cz)]),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
        ]
    )
    return rz @ ry @ rx


def _rotate_grid_and_vectors(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, velocity: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    flat = torch.stack([velocity[0].reshape(-1), velocity[1].reshape(-1), velocity[2].reshape(-1)], dim=0)
    rotated = rotation.T @ flat
    return rotated.reshape(3, *x.shape)


def random_flow_field(
    size: int = 32,
    device: torch.device | str = "cpu",
    return_mask: bool = False,
    peak_velocity_range: tuple[float, float] = (0.45, 1.60),
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic 3D velocity field with varied geometry.

    Returns:
        Tensor with shape [3, D, H, W] containing vx, vy, vz in m/s. If
        ``return_mask`` is True, also returns the binary vessel mask with shape
        [D, H, W].
    """
    x, y, z = make_grid(size, device)
    rotation = _random_rotation(device)
    coords = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=0)
    xr_flat, yr_flat, zr_flat = (rotation @ coords).reshape(3, *x.shape)
    x, y, z = xr_flat, yr_flat, zr_flat

    base_radius = torch.empty((), device=device).uniform_(0.38, 0.68)
    curvature = torch.empty((), device=device).uniform_(0.0, 0.26)
    phase_x = torch.empty((), device=device).uniform_(-math.pi, math.pi)
    phase_y = torch.empty((), device=device).uniform_(-math.pi, math.pi)
    freq_x = torch.empty((), device=device).uniform_(0.8, 2.4)
    freq_y = torch.empty((), device=device).uniform_(0.8, 2.4)
    center_x = curvature * torch.sin(freq_x * z + phase_x)
    center_y = curvature * torch.cos(freq_y * z + phase_y)
    xr = x - center_x
    yr = y - center_y
    r = torch.sqrt(xr.square() + yr.square() + 1e-8)

    stenosis_z = torch.empty((), device=device).uniform_(-0.45, 0.45)
    stenosis_width = torch.empty((), device=device).uniform_(0.18, 0.45)
    stenosis_depth = torch.empty((), device=device).uniform_(0.0, 0.40)
    radius = base_radius * (1.0 - stenosis_depth * torch.exp(-((z - stenosis_z).square()) / (2.0 * stenosis_width.square())))
    vessel = (r <= radius).float()

    vmax = torch.empty((), device=device).uniform_(0.8, 1.6)
    area_speedup = (base_radius / radius).square().clamp(max=3.0)
    profile = vmax * area_speedup * torch.clamp(1.0 - (r / radius).square(), min=0.0)

    swirl = torch.empty((), device=device).uniform_(-0.35, 0.35)
    jet_strength = torch.empty((), device=device).uniform_(0.0, 1.0)
    jet_x = torch.empty((), device=device).uniform_(-0.35, 0.35) * base_radius
    jet_y = torch.empty((), device=device).uniform_(-0.35, 0.35) * base_radius
    jet_width = torch.empty((), device=device).uniform_(0.10, 0.22)
    jet = jet_strength * torch.exp(-((x - jet_x).square() + (y - jet_y).square()) / (2.0 * jet_width.square()))

    branch_gate = torch.sigmoid(18.0 * (z - torch.empty((), device=device).uniform_(0.05, 0.45)))
    branch_angle = torch.empty((), device=device).uniform_(-0.55, 0.55)
    branch_strength = torch.empty((), device=device).uniform_(0.0, 0.35)

    pulse_phase = torch.empty((), device=device).uniform_(0.0, 2.0 * math.pi)
    pulse = 0.75 + 0.25 * torch.sin(math.pi * (z + 1.0) + pulse_phase)

    # Analytical derivatives of the centerline curve to define flow direction tangent
    dx_dz = curvature * freq_x * torch.cos(freq_x * z + phase_x)
    dy_dz = -curvature * freq_y * torch.sin(freq_y * z + phase_y)

    # Normalize the tangent vector to preserve velocity profile magnitude
    tangent_len = torch.sqrt(1.0 + dx_dz.square() + dy_dz.square())
    tx = dx_dz / tangent_len
    ty = dy_dz / tangent_len
    tz = 1.0 / tangent_len

    # Core axial flow following the curved centerline
    vz_main = (profile + jet) * pulse

    vz = vz_main * tz * vessel
    vx = (vz_main * tx - swirl * yr * profile + branch_strength * branch_gate * torch.sin(branch_angle) * profile) * vessel
    vy = (vz_main * ty + swirl * xr * profile + branch_strength * branch_gate * torch.cos(branch_angle) * profile) * vessel

    velocity = torch.stack([vx, vy, vz], dim=0)
    velocity = _rotate_grid_and_vectors(x, y, z, velocity, rotation)
    peak_velocity = torch.empty((), device=device).uniform_(*peak_velocity_range)
    velocity = velocity / velocity.abs().amax().clamp_min(1e-6) * peak_velocity
    if return_mask:
        return velocity, vessel
    return velocity
