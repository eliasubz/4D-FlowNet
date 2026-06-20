from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
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

    train_data = SyntheticFlowDataset(args.train_samples, noise_std=args.noise_std, seed=10)
    val_data = SyntheticFlowDataset(args.val_samples, noise_std=args.noise_std, seed=10000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FourDFlowNet(width=args.width, lr_blocks=args.lr_blocks, hr_blocks=args.hr_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for lr, hr in progress:
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(lr)
                loss = four_d_flow_loss(pred, hr, gradient_weight=args.gradient_weight)
                if args.divergence_weight > 0:
                    loss = loss + args.divergence_weight * divergence_l1(pred)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={running / len(train_loader):.5f}")
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

    checkpoint_path = args.output_dir / "4dflownet.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, checkpoint_path)
    print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
