from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import SyntheticFlowDataset, upsample_lr
from src.model import FourDFlowNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create blog-friendly comparison figures.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/4dflownet.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr-blocks", type=int, default=8)
    parser.add_argument("--hr-blocks", type=int, default=4)
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def speed(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v, dim=0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticFlowDataset(samples=args.sample + 1, seed=20000)
    lr, hr = dataset[args.sample]
    lr_batch = lr.unsqueeze(0).to(device)
    hr = hr.to(device)
    interp = upsample_lr(lr_batch, hr.shape[-1]).squeeze(0)

    model = FourDFlowNet(width=args.width, lr_blocks=args.lr_blocks, hr_blocks=args.hr_blocks).to(device)
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        pred = model(lr_batch).squeeze(0)
        pred_label = "Model"
    else:
        pred = interp
        pred_label = "Model missing"

    z = hr.shape[1] // 2
    panels = [
        ("Ground truth", speed(hr)[:, :, z]),
        ("Trilinear", speed(interp)[:, :, z]),
        (pred_label, speed(pred)[:, :, z]),
        ("Abs. error", speed(pred - hr)[:, :, z]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    vmax = panels[0][1].max().item()
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(image.detach().cpu(), origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="speed")
    out = args.output_dir / "speed_slice_comparison.png"
    fig.savefig(out, dpi=180)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
