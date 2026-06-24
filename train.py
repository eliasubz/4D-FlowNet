from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SyntheticFlowDataset, upsample_lr
from src.losses import four_d_flow_loss
from src.metrics import divergence_l1, endpoint_error, flow_rate_error, peak_velocity_error
from src.model import FourDFlowNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mini 4DFlowNet-style model on synthetic flow fields.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr-blocks", type=int, default=8)
    parser.add_argument("--hr-blocks", type=int, default=4)
    parser.add_argument("--gradient-weight", type=float, default=1e-3)
    parser.add_argument("--divergence-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    # Preprocessing mode.
    parser.add_argument(
        "--use-kspace-noise", action="store_true", default=False,
        help="Use paper-faithful k-space MRI simulation instead of spatial Gaussian noise.",
    )
    parser.add_argument("--snr-range", type=float, nargs=2, default=[14.0, 17.0], metavar=("MIN", "MAX"))
    parser.add_argument("--venc-range", type=float, nargs=2, default=[1.1, 3.0], metavar=("MIN", "MAX"))

    # Model upsampling mode.
    parser.add_argument(
        "--upsample-mode", choices=["subpixel", "trilinear"], default="subpixel",
        help="Upsampling strategy: 'subpixel' (learned, paper-faithful) or 'trilinear' (fixed).",
    )

    # Training extras.
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint in the output directory.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a specific model checkpoint to load.")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, hr_size: int = 32) -> dict[str, float]:
    model.eval()
    totals = {
        "model_mae": 0.0,
        "interp_mae": 0.0,
        "model_epe": 0.0,
        "interp_epe": 0.0,
        "model_peak_err": 0.0,
        "interp_peak_err": 0.0,
        "model_flow_err": 0.0,
        "interp_flow_err": 0.0,
        "model_div_l1": 0.0,
        "interp_div_l1": 0.0,
    }
    batches = 0

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        pred = model(lr)
        interp = upsample_lr(lr, hr_size)

        totals["model_mae"] += (pred - hr).abs().mean().item()
        totals["interp_mae"] += (interp - hr).abs().mean().item()
        totals["model_epe"] += endpoint_error(pred, hr).item()
        totals["interp_epe"] += endpoint_error(interp, hr).item()
        totals["model_peak_err"] += peak_velocity_error(pred, hr).item()
        totals["interp_peak_err"] += peak_velocity_error(interp, hr).item()
        totals["model_flow_err"] += flow_rate_error(pred, hr).item()
        totals["interp_flow_err"] += flow_rate_error(interp, hr).item()
        totals["model_div_l1"] += divergence_l1(pred).item()
        totals["interp_div_l1"] += divergence_l1(interp).item()
        batches += 1

    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs: dict = dict(noise_std=args.noise_std)
    if args.use_kspace_noise:
        dataset_kwargs.update(
            use_kspace_noise=True,
            snr_range=tuple(args.snr_range),
            venc_range=tuple(args.venc_range),
        )

    train_data = SyntheticFlowDataset(args.train_samples, seed=10, **dataset_kwargs)
    val_data = SyntheticFlowDataset(args.val_samples, seed=10000, **dataset_kwargs)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FourDFlowNet(
        width=args.width,
        lr_blocks=args.lr_blocks,
        hr_blocks=args.hr_blocks,
        upsample_mode=args.upsample_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_last_path = args.output_dir / "4dflownet_last.pt"
    checkpoint_best_path = args.output_dir / "4dflownet_best.pt"

    start_epoch = 1
    best_val_mae = float("inf")

    checkpoint_to_load = None
    if args.checkpoint:
        checkpoint_to_load = args.checkpoint
    elif args.resume and checkpoint_last_path.exists():
        checkpoint_to_load = checkpoint_last_path

    if checkpoint_to_load:
        print(f"Loading checkpoint from {checkpoint_to_load}...")
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and not args.checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and not args.checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scheduler.T_max = args.epochs
        if "epoch" in checkpoint and not args.checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_val_mae" in checkpoint:
            best_val_mae = checkpoint["best_val_mae"]
            print(f"Resumed from epoch {start_epoch}. Best Val MAE so far: {best_val_mae:.5f}")

    noise_mode = "kspace" if args.use_kspace_noise else "spatial"
    print(
        f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
        f"noise={noise_mode} upsample={args.upsample_mode}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for lr, hr in progress:
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(lr)
                loss = four_d_flow_loss(pred, hr, gradient_weight=args.gradient_weight)
                if args.divergence_weight > 0:
                    loss = loss + args.divergence_weight * divergence_l1(pred)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        val_mae = metrics["model_mae"]
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            print(f"New best model found at epoch {epoch}! Val MAE: {best_val_mae:.5f}")

        # Save checkpoints
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
            "metrics": metrics,
        }
        torch.save(checkpoint_payload, checkpoint_last_path)
        if is_best:
            torch.save(checkpoint_payload, checkpoint_best_path)

        print(f"epoch={epoch} train_loss={running / len(train_loader):.5f} lr={scheduler.get_last_lr()[0]:.2e}")
        print(
            "  model:  "
            f"mae={metrics['model_mae']:.5f} epe={metrics['model_epe']:.5f} "
            f"peak_err={100 * metrics['model_peak_err']:.2f}% "
            f"flow_err={100 * metrics['model_flow_err']:.2f}% "
            f"div_l1={metrics['model_div_l1']:.5f}"
        )
        print(
            "  interp: "
            f"mae={metrics['interp_mae']:.5f} epe={metrics['interp_epe']:.5f} "
            f"peak_err={100 * metrics['interp_peak_err']:.2f}% "
            f"flow_err={100 * metrics['interp_flow_err']:.2f}% "
            f"div_l1={metrics['interp_div_l1']:.5f}"
        )

    # Load best checkpoint for final model saving
    if checkpoint_best_path.exists():
        print(f"Loading best checkpoint from {checkpoint_best_path} for final model saving...")
        best_checkpoint = torch.load(checkpoint_best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

    checkpoint_path = args.output_dir / "4dflownet.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, checkpoint_path)
    print(f"saved final model to {checkpoint_path}")


if __name__ == "__main__":
    main()
