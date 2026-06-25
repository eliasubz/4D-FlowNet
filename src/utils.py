from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SyntheticFlowDataset, upsample_lr
from .losses import four_d_flow_loss
from .metrics import (
    divergence,
    divergence_l1,
    endpoint_error,
    flow_rate_error,
    peak_velocity_error,
)
from .model import FourDFlowNet


def checkerboard_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """3D factor-2 checkerboard score based on residual parity bias."""
    residual = pred - target
    if mask is not None:
        residual = residual * mask

    parity_means = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                block = residual[..., dz::2, dy::2, dx::2]
                if mask is not None:
                    block_mask = mask[..., dz::2, dy::2, dx::2]
                    denom = block_mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
                    parity_means.append(block.sum(dim=(-3, -2, -1)) / denom)
                else:
                    parity_means.append(block.mean(dim=(-3, -2, -1)))

    parity_means = torch.stack(parity_means, dim=0)
    parity_spread = parity_means.std(dim=0).mean()

    if mask is not None:
        residual_scale = residual.abs().sum() / mask.sum().clamp_min(1.0)
    else:
        residual_scale = residual.abs().mean()
    return parity_spread / residual_scale.clamp_min(eps)


def fluid_mask_from_hr(hr: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    speed = torch.linalg.vector_norm(hr, dim=1, keepdim=True)
    return (speed > threshold).float()


@torch.no_grad()
def evaluate_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    metric_names = ["mae", "epe", "peak", "flow", "div", "checker"]
    prefixes = ["", "tri_", "cub_", "sinc_"]
    batch_metrics = {f"{prefix}{name}": [] for prefix in prefixes for name in metric_names}

    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        mask = fluid_mask_from_hr(hr)

        interp_tri = upsample_lr(lr, hr.shape[-1], mode="trilinear")
        interp_cub = upsample_lr(lr, hr.shape[-1], mode="tricubic")
        interp_snc = upsample_lr(lr, hr.shape[-1], mode="sinc")

        candidates = {
            "": pred,
            "tri_": interp_tri,
            "cub_": interp_cub,
            "sinc_": interp_snc,
        }

        for prefix, out in candidates.items():
            batch_metrics[f"{prefix}mae"].append((out - hr).abs().mean().item())
            batch_metrics[f"{prefix}epe"].append(endpoint_error(out, hr).item())
            batch_metrics[f"{prefix}peak"].append(peak_velocity_error(out, hr).item())
            batch_metrics[f"{prefix}flow"].append(flow_rate_error(out, hr).item())
            batch_metrics[f"{prefix}div"].append(divergence_l1(out).item())
            batch_metrics[f"{prefix}checker"].append(checkerboard_index(out, hr, mask=mask).item())

    means = {key: float(np.mean(vals)) for key, vals in batch_metrics.items()}
    stds = {key: float(np.std(vals)) for key, vals in batch_metrics.items()}
    return means, stds


def save_checkpoint(
    run_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: list[dict[str, float]],
    metrics: dict[str, float],
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    payload = {
        "run_name": run_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "metrics": metrics,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_dir / f"{run_name}_last.pt")
    if is_best:
        torch.save(payload, checkpoint_dir / f"{run_name}_best.pt")


def save_fixed_predictions(
    run_name: str,
    model: torch.nn.Module,
    device: torch.device,
    prediction_dir: Path,
    seed: int = 2026,
    samples: int = 4,
) -> None:
    ds = SyntheticFlowDataset(samples=samples, seed=seed, use_kspace_noise=True, snr_range=(14.0, 17.0))
    loader = DataLoader(ds, batch_size=samples, shuffle=False)
    lr, hr = next(iter(loader))
    model.eval()
    with torch.no_grad():
        pred = model(lr.to(device)).cpu()

    prediction_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"lr": lr.cpu(), "hr": hr.cpu(), "pred": pred, "seed": seed, "run_name": run_name},
        prediction_dir / f"{run_name}_fixed_predictions.pt",
    )


def axis_size(volume: torch.Tensor, component: int) -> int:
    # volume shape: [3, D, H, W]
    _, d, h, w = volume.shape
    return [d, h, w][component]


def velocity_slice_at_fraction(volume: torch.Tensor, component: int, frac: float) -> torch.Tensor:
    _, d, h, w = volume.shape

    if component == 0:
        idx = int(round(frac * (d - 1)))
        return volume[0, idx, :, :]

    if component == 1:
        idx = int(round(frac * (h - 1)))
        return volume[1, :, idx, :]

    if component == 2:
        idx = int(round(frac * (w - 1)))
        return volume[2, :, :, idx]

    raise ValueError(component)


def scalar_slice_at_fraction(field: torch.Tensor, component: int, frac: float) -> torch.Tensor:
    # field shape: [D, H, W]
    d, h, w = field.shape

    if component == 0:
        idx = int(round(frac * (d - 1)))
        return field[idx, :, :]

    if component == 1:
        idx = int(round(frac * (h - 1)))
        return field[:, idx, :]

    if component == 2:
        idx = int(round(frac * (w - 1)))
        return field[:, :, idx]

    raise ValueError(component)


def scalar_slice_at(field: torch.Tensor, component: int, idx: int) -> torch.Tensor:
    # field shape: [D, H, W]
    d, h, w = field.shape

    if component == 0:
        idx = min(max(int(idx), 0), d - 1)
        return field[idx, :, :]

    if component == 1:
        idx = min(max(int(idx), 0), h - 1)
        return field[:, idx, :]

    if component == 2:
        idx = min(max(int(idx), 0), w - 1)
        return field[:, :, idx]

    raise ValueError(f"unknown component: {component}")


def edge_score_2d(img: torch.Tensor) -> torch.Tensor:
    img = img.float()
    dx = img[:, 1:] - img[:, :-1]
    dy = img[1:, :] - img[:-1, :]
    return dx.abs().mean() + dy.abs().mean()


def strongest_slice_index(hr: torch.Tensor, component: int, margin: int = 2) -> tuple[int, float]:
    # Pick from HR speed magnitude so the selected slice tends to contain vessel walls.
    speed = torch.linalg.vector_norm(hr, dim=0)
    d, h, w = speed.shape
    axis_size_val = [d, h, w][component]

    lo = min(margin, axis_size_val - 1)
    hi = max(lo + 1, axis_size_val - margin)

    best_idx = lo
    best_score = -float("inf")

    for idx in range(lo, hi):
        img = scalar_slice_at(speed, component, idx)
        score = float(edge_score_2d(img).item())

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score


def divergence_volume(volume: torch.Tensor, mask_threshold: float = 1e-4) -> torch.Tensor:
    # volume shape: [3, D, H, W]
    v = volume.unsqueeze(0)
    div = divergence(v).squeeze(0).detach().cpu()

    speed = torch.linalg.vector_norm(volume.detach().cpu(), dim=0)
    mask = speed[1:-1, 1:-1, 1:-1] > mask_threshold

    return div.masked_fill(~mask, float("nan"))


def finite_minmax(images: list[torch.Tensor], fallback: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    values = []

    for img in images:
        flat = img.reshape(-1)
        flat = flat[torch.isfinite(flat)]
        if flat.numel() > 0:
            values.append(flat)

    if not values:
        return fallback

    combined = torch.cat(values)
    vmin = float(combined.min().item())
    vmax = float(combined.max().item())

    if abs(vmax - vmin) < 1e-8:
        pad = max(abs(vmax), 1.0) * 0.05
        vmin -= pad
        vmax += pad

    return vmin, vmax


def fluid_mask_from_volume(volume: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    speed = torch.linalg.vector_norm(volume, dim=0)
    return speed > threshold


@torch.no_grad()
def make_2d_ablation_table(
    trilinear_model: torch.nn.Module,
    subpixel_no_icnr_model: torch.nn.Module,
    subpixel_icnr_model: torch.nn.Module,
    device: torch.device,
    figure_dir: Path,
    seed: int = 999,
    sample_idx: int = 3,
    save_name: str = "2d_ablation_table.png",
):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    ds = SyntheticFlowDataset(samples=max(sample_idx + 1, 4), use_kspace_noise=True, seed=seed)
    lr, hr = ds[sample_idx]

    lr = lr.cpu()
    hr = hr.cpu()
    lr_batch = lr.unsqueeze(0).to(device)

    trilinear_model.eval()
    subpixel_no_icnr_model.eval()
    subpixel_icnr_model.eval()

    linear = upsample_lr(lr_batch, hr.shape[-1], mode="trilinear").squeeze(0).cpu()
    pred_trilinear = trilinear_model(lr_batch).squeeze(0).cpu()
    pred_pixelshuffle = subpixel_no_icnr_model(lr_batch).squeeze(0).cpu()
    pred_pixelshuffle_icnr = subpixel_icnr_model(lr_batch).squeeze(0).cpu()

    mask = fluid_mask_from_volume(hr)
    pred_trilinear = pred_trilinear * mask
    pred_pixelshuffle = pred_pixelshuffle * mask
    pred_pixelshuffle_icnr = pred_pixelshuffle_icnr * mask

    columns = [
        ("LR", lr[:3]),
        ("HR", hr),
        ("Linear", linear),
        ("4DFlowNet\ntrilinear", pred_trilinear),
        ("4DFlowNet\nPixelShuffle", pred_pixelshuffle),
        ("4DFlowNet\nPixelShuffle+ICNR", pred_pixelshuffle_icnr),
    ]

    cm_per_m = 100.0

    best_indices = {}
    best_fractions = {}
    for component in range(3):
        idx, score = strongest_slice_index(hr, component, margin=2)
        size = axis_size(hr, component)
        frac = idx / (size - 1)
        best_indices[component] = idx
        best_fractions[component] = frac
        print(f"component {component}: HR slice={idx}, frac={frac:.3f}, edge_score={score:.5f}")

    image_grid = []
    for component in range(3):
        row = []
        for _, volume in columns:
            frac = best_fractions[component]
            row.append(velocity_slice_at_fraction(volume, component, frac) * cm_per_m)
        image_grid.append(row)

    div_row = []
    div_frac = best_fractions[2]
    for _, volume in columns:
        div = divergence_volume(volume)
        div_row.append(scalar_slice_at_fraction(div, component=2, frac=div_frac))
    image_grid.append(div_row)

    error_row = []
    for name, volume in columns:
        if name in {"LR", "HR"}:
            error_row.append(torch.zeros_like(scalar_slice_at(hr[0], 2, best_indices[2])))
        else:
            speed_error = torch.linalg.vector_norm(volume - hr, dim=0) * cm_per_m
            error_row.append(scalar_slice_at(speed_error, 2, best_indices[2]))
    image_grid.append(error_row)

    row_ranges = [finite_minmax(row) for row in image_grid]
    row_labels = [
        "$V_x$\n(cm/s)",
        "$V_y$\n(cm/s)",
        "$V_z$\n(cm/s)",
        "divergence\n(1/s)",
        "velocity\nerror\n(cm/s)",
    ]

    n_rows = 5
    n_cols = len(columns)
    fig = plt.figure(figsize=(11, 8.4), dpi=220)

    gs = gridspec.GridSpec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.035],
        wspace=0.0,
        hspace=0.0,
        left=0.075,
        right=0.965,
        top=0.92,
        bottom=0.06,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    cbar_axes = [fig.add_subplot(gs[r, n_cols]) for r in range(n_rows)]

    for r in range(n_rows):
        vmin, vmax = row_ranges[r]
        last_im = None

        for c in range(n_cols):
            ax = axes[r][c]
            img = image_grid[r][c]

            if r == n_rows - 2:
                cmap = "terrain"
            elif r == n_rows - 1:
                cmap = "gnuplot"
            else:
                cmap = "jet"

            last_im = ax.imshow(
                img,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.55)
                spine.set_color("#222")

            if r == 0:
                ax.set_title(columns[c][0], fontsize=8.5, fontweight="bold", pad=3)

            if c == 0:
                ax.set_ylabel(
                    row_labels[r],
                    fontsize=8.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=30,
                    va="center",
                )

        cbar = fig.colorbar(last_im, cax=cbar_axes[r])
        cbar.ax.tick_params(labelsize=6, length=2)
        cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        "2D ablation table: velocity-component slices and divergence",
        fontsize=11,
        fontweight="bold",
        y=0.965,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / save_name
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    print(f"saved: {out_path}")

    return fig


def train_model(
    upsample_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    prediction_dir: Path,
    epochs: int = 15,
    train_samples: int = 1024,
    val_samples: int = 128,
    icnr: bool = False,
    batch_size: int = 8,
) -> tuple[list[dict[str, float]], FourDFlowNet]:
    train_dataset = SyntheticFlowDataset(train_samples, seed=10, use_kspace_noise=True, snr_range=(14.0, 17.0))
    val_dataset = SyntheticFlowDataset(val_samples, seed=10000, use_kspace_noise=True, snr_range=(14.0, 17.0))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    run_name = f"{upsample_mode}_icnr" if icnr else f"{upsample_mode}_no_icnr"
    checkpoint_path = checkpoint_dir / f"{run_name}_last.pt"

    start_epoch = 1
    history: list[dict[str, float]] = []
    best_val_mae = float("inf")

    model = FourDFlowNet(
        width=24,
        lr_blocks=4,
        hr_blocks=2,
        upsample_mode=upsample_mode,
        use_icnr=icnr,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if checkpoint_path.exists():
        print(f"Resuming {run_name} from checkpoint {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scheduler.T_max = epochs
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        if history:
            best_val_mae = min(h["mae"] for h in history)
            print(f"Loaded history with {len(history)} epochs. Best Val MAE so far: {best_val_mae:.5f}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(lr)
                loss = four_d_flow_loss(pred, hr, gradient_weight=1e-3)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        val_metrics, val_stds = evaluate_all(model, val_loader, device)
        val_metrics["train_loss"] = epoch_loss
        val_metrics["epoch"] = epoch
        val_metrics["upsample_mode"] = upsample_mode
        val_metrics["icnr"] = icnr

        for key, value in val_stds.items():
            val_metrics[f"{key}_std"] = value

        history.append(val_metrics)

        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            print(f"New best model found at epoch {epoch}! Val MAE: {best_val_mae:.5f}")

        save_checkpoint(
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            history=history,
            metrics=val_metrics,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best,
        )

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"{upsample_mode} icnr={icnr} | "
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {epoch_loss:.5f} | "
                f"Val MAE: {val_metrics['mae']:.5f} | "
                f"Peak Err: {100 * val_metrics['peak']:.1f}% | "
                f"Div L1: {val_metrics['div']:.5f}"
            )

    best_path = checkpoint_dir / f"{run_name}_best.pt"
    if best_path.exists():
        print(f"Loading best model from {best_path} for returning...")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        history = checkpoint["history"]

    save_fixed_predictions(run_name, model, device, prediction_dir)
    return history, model
