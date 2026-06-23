from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EPOCH_RE = re.compile(
    r"(?P<name>\S+) epoch=(?P<epoch>\d+) loss=(?P<loss>[0-9.]+) "
    r"matched_mae=(?P<model_mae>[0-9.]+)/(?P<interp_mae>[0-9.]+) "
    r"matched_peak=(?P<model_peak>[0-9.]+)/(?P<interp_peak>[0-9.]+)% "
    r"matched_flow=(?P<model_flow>[0-9.]+)/(?P<interp_flow>[0-9.]+)%"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean CSVs and plots from A100 experiment outputs.")
    parser.add_argument("--summary", type=Path, default=Path("outputs/summary.csv"))
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to a training log file (optional, for epoch-level plots).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report"))
    return parser.parse_args()


def add_improvements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metric_pairs = [
        ("matched_mae", "matched_model_mae", "matched_interp_mae", 1.0),
        ("matched_peak_pct", "matched_model_peak_err", "matched_interp_peak_err", 100.0),
        ("matched_flow_pct", "matched_model_flow_err", "matched_interp_flow_err", 100.0),
        ("noisy_mae", "noisy_model_mae", "noisy_interp_mae", 1.0),
        ("noisy_peak_pct", "noisy_model_peak_err", "noisy_interp_peak_err", 100.0),
        ("noisy_flow_pct", "noisy_model_flow_err", "noisy_interp_flow_err", 100.0),
    ]
    for label, model_col, interp_col, scale in metric_pairs:
        out[f"{label}_model"] = out[model_col] * scale
        out[f"{label}_trilinear"] = out[interp_col] * scale
        out[f"{label}_relative_improvement_pct"] = (1.0 - out[model_col] / out[interp_col]) * 100.0
    return out


def parse_epoch_log(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text(errors="replace").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append(
            {
                "name": row["name"],
                "epoch": int(row["epoch"]),
                "train_loss": float(row["loss"]),
                "matched_model_mae": float(row["model_mae"]),
                "matched_interp_mae": float(row["interp_mae"]),
                "matched_model_peak_pct": float(row["model_peak"]),
                "matched_interp_peak_pct": float(row["interp_peak"]),
                "matched_model_flow_pct": float(row["model_flow"]),
                "matched_interp_flow_pct": float(row["interp_flow"]),
            }
        )
    return pd.DataFrame(rows)


def plot_final(clean: pd.DataFrame, output: Path) -> None:
    names = clean["name"].tolist()
    x = range(len(clean))
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    axes = axes.ravel()
    specs = [
        ("matched_mae_model", "matched_mae_trilinear", "Matched MAE"),
        ("matched_peak_pct_model", "matched_peak_pct_trilinear", "Matched peak error (%)"),
        ("matched_flow_pct_model", "matched_flow_pct_trilinear", "Matched flow error (%)"),
        ("noisy_mae_model", "noisy_mae_trilinear", "Noisy MAE"),
        ("noisy_peak_pct_model", "noisy_peak_pct_trilinear", "Noisy peak error (%)"),
        ("noisy_flow_pct_model", "noisy_flow_pct_trilinear", "Noisy flow error (%)"),
    ]
    for ax, (model_col, interp_col, title) in zip(axes, specs):
        ax.bar([i - 0.18 for i in x], clean[model_col], width=0.36, label="model")
        ax.bar([i + 0.18 for i in x], clean[interp_col], width=0.36, label="trilinear")
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_epochs(epochs: pd.DataFrame, output: Path) -> None:
    if epochs.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    specs = [
        ("matched_model_mae", "matched_interp_mae", "Matched MAE"),
        ("matched_model_peak_pct", "matched_interp_peak_pct", "Matched peak error (%)"),
        ("matched_model_flow_pct", "matched_interp_flow_pct", "Matched flow error (%)"),
    ]
    for ax, (model_col, interp_col, title) in zip(axes, specs):
        for name, group in epochs.groupby("name"):
            ax.plot(group["epoch"], group[model_col], marker="o", markersize=2.5, label=name)
        first = epochs.groupby("epoch")[interp_col].mean().reset_index()
        ax.plot(first["epoch"], first[interp_col], color="black", linestyle="--", linewidth=1.2, label="trilinear avg")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    clean = add_improvements(summary)
    clean_path = args.output_dir / "a100_quick_summary_clean.csv"
    clean.to_csv(clean_path, index=False)

    epoch_log = parse_epoch_log(args.log) if args.log else pd.DataFrame()
    epoch_path = args.output_dir / "a100_quick_epoch_log.csv"
    epoch_log.to_csv(epoch_path, index=False)

    final_plot = args.output_dir / "a100_quick_final_metrics.png"
    epoch_plot = args.output_dir / "a100_quick_training_curves.png"
    plot_final(clean, final_plot)
    plot_epochs(epoch_log, epoch_plot)

    print(clean_path.resolve())
    print(epoch_path.resolve())
    print(final_plot.resolve())
    if epoch_plot.exists():
        print(epoch_plot.resolve())


if __name__ == "__main__":
    main()
