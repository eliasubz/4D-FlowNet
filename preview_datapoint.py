from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import SyntheticFlowDataset, upsample_lr
from src.synthetic_flows import make_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview one random synthetic 4D flow datapoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/random_datapoint.png"))
    return parser.parse_args()


def speed(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v, dim=0)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticFlowDataset(samples=args.sample + 1, seed=args.seed)
    lr_input, hr = dataset[args.sample]
    lr_velocity = lr_input[:3].unsqueeze(0)
    interp = upsample_lr(lr_velocity, hr.shape[-1]).squeeze(0)

    hr_speed = speed(hr)
    interp_speed = speed(interp)
    error = speed(interp - hr)
    z = hr.shape[1] // 2
    y = hr.shape[2] // 2

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    axes = {
        "hr_xy": fig.add_subplot(grid[0, 0]),
        "field3d": fig.add_subplot(grid[0, 1], projection="3d"),
        "err_xy": fig.add_subplot(grid[0, 2]),
        "hr_xz": fig.add_subplot(grid[1, 0]),
        "quiver": fig.add_subplot(grid[1, 1]),
        "hist": fig.add_subplot(grid[1, 2]),
    }

    vmax = hr_speed.max().item()
    image_specs = [
        ("hr_xy", "HR speed, axial slice", hr_speed[z], "viridis", 0.0, vmax),
        ("err_xy", "Interpolation error", error[z], "magma", 0.0, error.max().item()),
        ("hr_xz", "HR speed, sagittal slice", hr_speed[:, y, :], "viridis", 0.0, vmax),
    ]

    for key, title, image, cmap, vmin, vmax_i in image_specs:
        im = axes[key].imshow(image.detach().cpu(), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_i)
        axes[key].set_title(title)
        axes[key].set_xticks([])
        axes[key].set_yticks([])
        fig.colorbar(im, ax=axes[key], fraction=0.046, pad=0.04)

    x, yy, zz = make_grid(hr.shape[-1])
    mask = hr_speed > 0.12
    step3d = 4
    sampled = torch.zeros_like(mask, dtype=torch.bool)
    sampled[::step3d, ::step3d, ::step3d] = True
    mask = mask & sampled
    x3 = x[mask].detach().cpu()
    y3 = yy[mask].detach().cpu()
    z3 = zz[mask].detach().cpu()
    u3 = hr[0][mask].detach().cpu()
    v3 = hr[1][mask].detach().cpu()
    w3 = hr[2][mask].detach().cpu()
    color3 = hr_speed[mask].detach().cpu()

    axes["field3d"].quiver(x3, y3, z3, u3, v3, w3, length=0.16, normalize=True, colors=plt.cm.viridis(color3 / color3.max()))
    axes["field3d"].set_title("3D velocity field")
    axes["field3d"].set_xlim(-1, 1)
    axes["field3d"].set_ylim(-1, 1)
    axes["field3d"].set_zlim(-1, 1)
    axes["field3d"].set_xticks([])
    axes["field3d"].set_yticks([])
    axes["field3d"].set_zticks([])
    axes["field3d"].view_init(elev=22, azim=-58)

    step = 2
    vx = hr[0, z, ::step, ::step].detach().cpu()
    vy = hr[1, z, ::step, ::step].detach().cpu()
    axes["quiver"].imshow(hr_speed[z].detach().cpu(), origin="lower", cmap="Greys", alpha=0.45)
    axes["quiver"].quiver(vx, vy, color="tab:red", angles="xy", scale_units="xy", scale=0.35, width=0.005)
    axes["quiver"].set_title("In-plane velocity arrows")
    axes["quiver"].set_xticks([])
    axes["quiver"].set_yticks([])

    nonzero = hr_speed[hr_speed > 0.02].detach().cpu()
    axes["hist"].hist(nonzero.numpy(), bins=30, color="steelblue")
    axes["hist"].set_title("Speed distribution")
    axes["hist"].set_xlabel("normalized speed")
    axes["hist"].set_ylabel("voxels")

    fig.suptitle(f"Random synthetic 4DFlowNet-mini datapoint | seed={args.seed}, sample={args.sample}")
    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
