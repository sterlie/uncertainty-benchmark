"""Bar plot of diagnosis counts per PC1 bin for the isic_pc1 task.

Usage (from repo root):
    python -m src.utils.plot_pc1_bins
    python -m src.utils.plot_pc1_bins dataset.pc1.max_dist=8 seed=11112
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from src.datasets.isic import build_isic_pc1_splits, compute_isic_pc1
from src.experiments.datasets.isic_adapter import _read_merged_table

# Fixed categorical order (reference data-viz palette, slots 1-6)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _counts_per_bin(frame: pd.DataFrame, classes, n_bins: int) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    counts = pd.crosstab(frame["pc1_bin"], frame["diagnosis"])
    return counts.reindex(index=range(n_bins), columns=classes, fill_value=0)


def main(overrides):
    config_dir = os.path.abspath("config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["dataset=isic", "experiment=isic_pc1", f"data.root={os.path.abspath('data')}", *overrides],
        )

    pc1_cfg = cfg.dataset.pc1
    pca_size = int(pc1_cfg.get("pca_image_size", 64))
    data_root = str(cfg.dataset.data_root)
    cache_csv = pc1_cfg.get("cache_csv", os.path.join(data_root, f"pc1_scores_{pca_size}px.csv"))

    df = _read_merged_table(cfg)
    df = df.join(compute_isic_pc1(df, image_size=pca_size, cache_csv=cache_csv))
    classes = [str(c) for c in cfg.dataset.selected_classes]
    df["diagnosis"] = [classes[int(np.argmax(label))] for label in df["label"]]

    n_bins = int(pc1_cfg.n_bins)
    train_df, val_df, test_frames = build_isic_pc1_splits(
        df,
        n_bins=n_bins,
        max_dist=float(pc1_cfg.max_dist),
        val_ratio=float(pc1_cfg.val_ratio),
        test_per_bin=int(pc1_cfg.test_per_bin),
        binning=str(pc1_cfg.binning),
        seed=int(cfg.seed),
    )

    panels = {
        f"Training + validation (n={len(train_df) + len(val_df)})": _counts_per_bin(
            pd.concat([train_df, val_df]), classes, n_bins
        ),
        f"Test (n={sum(len(d) for d in test_frames.values())})": _counts_per_bin(
            pd.concat(test_frames.values()), classes, n_bins
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(n_bins)
    bin_labels = [f"bin_{b + 1}" for b in range(n_bins)]
    for ax, (title, counts) in zip(axes, panels.items()):
        bottom = np.zeros(n_bins)
        for cls, color in zip(classes, COLORS):
            ax.bar(x, counts[cls].values, bottom=bottom, width=0.6, color=color,
                   edgecolor="white", linewidth=1.5, label=cls)
            bottom += counts[cls].values
        for xi, total in zip(x, bottom):
            ax.text(xi, total, f"{int(total)}", ha="center", va="bottom", fontsize=9, color="#52514e")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("PC1 bin (dark → bright)")
        ax.set_ylabel("Number of images")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, bottom.max() * 1.1)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Diagnosis", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"isic_pc1: diagnoses per PC1 bin (max_dist={pc1_cfg.max_dist}, seed={cfg.seed})")
    fig.tight_layout()

    out_dir = os.path.join("src", "plots")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"pc1_diagnosis_per_bin.{ext}"), bbox_inches="tight", dpi=150)
    print({"plot": os.path.join(out_dir, "pc1_diagnosis_per_bin.png")})
    for title, counts in panels.items():
        print(title)
        print(counts.set_axis(bin_labels).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
