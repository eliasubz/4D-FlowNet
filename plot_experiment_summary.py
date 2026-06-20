from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A100 experiment summary metrics.")
    parser.add_argument("--summary", type=Path, default=Path("outputs/experiments/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/summary.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)

    names = df["name"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()

    plots = [
        ("matched_model_mae", "matched_interp_mae", "MAE"),
        ("matched_model_peak_err", "matched_interp_peak_err", "Peak velocity error", 100.0),
        ("matched_model_flow_err", "matched_interp_flow_err", "Flow-rate error", 100.0),
        ("noisy_model_mae", "noisy_interp_mae", "Noisy-val MAE"),
        ("seconds_per_epoch", None, "Seconds / epoch"),
        ("peak_memory_gb", None, "Peak memory (GB)"),
    ]

    x = range(len(df))
    for ax, spec in zip(axes, plots):
        model_col, interp_col, title = spec[:3]
        scale = spec[3] if len(spec) > 3 else 1.0
        ax.bar([i - 0.18 for i in x], df[model_col] * scale, width=0.36, label="model")
        if interp_col:
            ax.bar([i + 0.18 for i in x], df[interp_col] * scale, width=0.36, label="trilinear")
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
