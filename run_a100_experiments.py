from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SyntheticFlowDataset, upsample_lr
from src.losses import four_d_flow_loss
from src.metrics import divergence_l1, endpoint_error, flow_rate_error, peak_velocity_error
from src.model import FourDFlowNet


@dataclass
class Experiment:
    name: str
    width: int
    lr_blocks: int
    hr_blocks: int
    gradient_weight: float
    divergence_weight: float
    train_noise: float
    train_samples: int
    epochs: int
    batch_size: int
    lr: float = 2e-4
    use_kspace_noise: bool = False
    upsample_mode: str = "trilinear"


def get_experiments(preset: str) -> list[Experiment]:
    if preset == "smoke":
        return [
            Experiment("smoke_mse", 16, 2, 1, 0.0, 0.0, 0.03, 128, 2, 4),
            Experiment("smoke_vg", 16, 2, 1, 1e-3, 0.0, 0.03, 128, 2, 4),
        ]

    if preset == "a100_quick":
        return [
            Experiment("medium_mse", 32, 4, 2, 0.0, 0.0, 0.03, 2048, 20, 16),
            Experiment("medium_vg", 32, 4, 2, 1e-3, 0.0, 0.03, 2048, 20, 16),
            Experiment("medium_vg_div", 32, 4, 2, 1e-3, 1e-4, 0.03, 2048, 20, 16),
            Experiment("medium_vg_noisy", 32, 4, 2, 1e-3, 0.0, 0.08, 2048, 20, 16),
        ]

    if preset == "a100_main":
        return [
            Experiment("medium_mse", 32, 4, 2, 0.0, 0.0, 0.03, 4096, 40, 16),
            Experiment("medium_vg", 32, 4, 2, 1e-3, 0.0, 0.03, 4096, 40, 16),
            Experiment("medium_vg_div", 32, 4, 2, 1e-3, 1e-4, 0.03, 4096, 40, 16),
            Experiment("paperish_vg", 64, 8, 4, 1e-3, 0.0, 0.03, 4096, 35, 8),
            Experiment("paperish_vg_div", 64, 8, 4, 1e-3, 1e-4, 0.03, 4096, 35, 8),
            Experiment("paperish_vg_noisy", 64, 8, 4, 1e-3, 0.0, 0.08, 4096, 35, 8),
        ]

    if preset == "a100_upgrade":
        # 2x2 ablation: {spatial/kspace} x {trilinear/subpixel}
        return [
            Experiment("baseline_spatial",  32, 4, 2, 1e-3, 0.0, 0.03, 4096, 30, 16,
                        use_kspace_noise=False, upsample_mode="trilinear"),
            Experiment("kspace_only",       32, 4, 2, 1e-3, 0.0, 0.03, 4096, 30, 16,
                        use_kspace_noise=True,  upsample_mode="trilinear"),
            Experiment("subpixel_only",     32, 4, 2, 1e-3, 0.0, 0.03, 4096, 30, 16,
                        use_kspace_noise=False, upsample_mode="subpixel"),
            Experiment("full_paper",        32, 4, 2, 1e-3, 0.0, 0.03, 4096, 30, 16,
                        use_kspace_noise=True,  upsample_mode="subpixel"),
        ]

    if preset == "smoke_upgrade":
        # Quick smoke test for the 2x2 ablation.
        return [
            Experiment("baseline_spatial",  16, 2, 1, 1e-3, 0.0, 0.03, 128, 3, 4,
                        use_kspace_noise=False, upsample_mode="trilinear"),
            Experiment("kspace_only",       16, 2, 1, 1e-3, 0.0, 0.03, 128, 3, 4,
                        use_kspace_noise=True,  upsample_mode="trilinear"),
            Experiment("subpixel_only",     16, 2, 1, 1e-3, 0.0, 0.03, 128, 3, 4,
                        use_kspace_noise=False, upsample_mode="subpixel"),
            Experiment("full_paper",        16, 2, 1, 1e-3, 0.0, 0.03, 128, 3, 4,
                        use_kspace_noise=True,  upsample_mode="subpixel"),
        ]

    raise ValueError(f"unknown preset: {preset}")


def make_loader(
    samples: int,
    noise_std: float,
    seed: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    use_kspace_noise: bool = False,
) -> DataLoader:
    dataset = SyntheticFlowDataset(
        samples=samples, noise_std=noise_std, seed=seed,
        use_kspace_noise=use_kspace_noise,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, tag: str) -> dict[str, float]:
    model.eval()
    totals = {
        f"{tag}_model_mae": 0.0,
        f"{tag}_interp_mae": 0.0,
        f"{tag}_model_epe": 0.0,
        f"{tag}_interp_epe": 0.0,
        f"{tag}_model_peak_err": 0.0,
        f"{tag}_interp_peak_err": 0.0,
        f"{tag}_model_flow_err": 0.0,
        f"{tag}_interp_flow_err": 0.0,
        f"{tag}_model_div_l1": 0.0,
        f"{tag}_interp_div_l1": 0.0,
    }
    batches = 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        pred = model(lr)
        interp = upsample_lr(lr, hr.shape[-1])

        totals[f"{tag}_model_mae"] += F.l1_loss(pred, hr).item()
        totals[f"{tag}_interp_mae"] += F.l1_loss(interp, hr).item()
        totals[f"{tag}_model_epe"] += endpoint_error(pred, hr).item()
        totals[f"{tag}_interp_epe"] += endpoint_error(interp, hr).item()
        totals[f"{tag}_model_peak_err"] += peak_velocity_error(pred, hr).item()
        totals[f"{tag}_interp_peak_err"] += peak_velocity_error(interp, hr).item()
        totals[f"{tag}_model_flow_err"] += flow_rate_error(pred, hr).item()
        totals[f"{tag}_interp_flow_err"] += flow_rate_error(interp, hr).item()
        totals[f"{tag}_model_div_l1"] += divergence_l1(pred).item()
        totals[f"{tag}_interp_div_l1"] += divergence_l1(interp).item()
        batches += 1
    return {key: value / batches for key, value in totals.items()}


def train_one(exp: Experiment, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, float], list[dict[str, float]]]:
    out_dir = args.output_dir / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(exp.train_samples, exp.train_noise, 10, exp.batch_size, args.workers, True,
                               use_kspace_noise=exp.use_kspace_noise)
    val_clean = make_loader(args.val_samples, 0.0, 10000, exp.batch_size, args.workers, False,
                            use_kspace_noise=exp.use_kspace_noise)
    val_matched = make_loader(args.val_samples, exp.train_noise, 20000, exp.batch_size, args.workers, False,
                              use_kspace_noise=exp.use_kspace_noise)
    val_noisy = make_loader(args.val_samples, 0.10, 30000, exp.batch_size, args.workers, False,
                            use_kspace_noise=exp.use_kspace_noise)

    model = FourDFlowNet(
        width=exp.width, lr_blocks=exp.lr_blocks, hr_blocks=exp.hr_blocks,
        upsample_mode=exp.upsample_mode,
    ).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=exp.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    params = sum(p.numel() for p in model.parameters())
    history: list[dict[str, float]] = []
    best_score = float("inf")
    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, exp.epochs + 1):
        model.train()
        running = 0.0
        epoch_start = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"{exp.name} {epoch}/{exp.epochs}", leave=False)
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(lr)
                loss = four_d_flow_loss(pred, hr, gradient_weight=exp.gradient_weight)
                if exp.divergence_weight:
                    loss = loss + exp.divergence_weight * divergence_l1(pred)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": running / len(train_loader),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        row.update(evaluate(model, val_clean, device, "clean"))
        row.update(evaluate(model, val_matched, device, "matched"))
        row.update(evaluate(model, val_noisy, device, "noisy"))
        history.append(row)

        score = row["matched_model_flow_err"] + row["matched_model_peak_err"] + row["matched_model_mae"]
        if score < best_score:
            best_score = score
            checkpoint = {
                "model": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                "experiment": asdict(exp),
                "epoch": epoch,
                "metrics": row,
            }
            torch.save(checkpoint, out_dir / "best.pt")

        print(
            f"{exp.name} epoch={epoch:03d} "
            f"loss={row['train_loss']:.5f} "
            f"matched_mae={row['matched_model_mae']:.5f}/{row['matched_interp_mae']:.5f} "
            f"matched_peak={100 * row['matched_model_peak_err']:.2f}/{100 * row['matched_interp_peak_err']:.2f}% "
            f"matched_flow={100 * row['matched_model_flow_err']:.2f}/{100 * row['matched_interp_flow_err']:.2f}%"
        )

    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    final = history[-1].copy()
    final.update({f"config_{k}": v for k, v in asdict(exp).items() if k != "name"})
    final["name"] = exp.name
    final["params"] = float(params)
    final["total_seconds"] = elapsed
    final["peak_memory_gb"] = peak_gb
    final["seconds_per_epoch"] = elapsed / exp.epochs

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(exp), f, indent=2)

    return final, history


def write_summary(rows: list[dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A100-sized 4DFlowNet-mini experiment sweeps.")
    parser.add_argument(
        "--preset",
        choices=["smoke", "smoke_upgrade", "a100_quick", "a100_main", "a100_upgrade"],
        default="a100_quick",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--val-samples", type=int, default=384)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} preset={args.preset} output={args.output_dir}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    rows = []
    for exp in get_experiments(args.preset):
        final, _history = train_one(exp, args, device)
        rows.append(final)
        write_summary(rows, args.output_dir / "summary.csv")
        with (args.output_dir / "summary.json").open("w") as f:
            json.dump(rows, f, indent=2)

    print(f"saved {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
