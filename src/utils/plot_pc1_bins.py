"""Bar plot of diagnosis counts per PC1 bin of the test set for the isic_pc1 task.

Usage (from repo root):
    python -m src.utils.plot_pc1_bins
    python -m src.utils.plot_pc1_bins dataset.pc1.n_bins=6 dataset.pc1.bin_by=dist
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from src.experiments.datasets.isic_adapter import (
    _read_merged_table,
    _split_kfold,
    _split_single,
    build_isic_pc1_frames,
)

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
    df = _read_merged_table(cfg)
    if bool(cfg.dataset.get("cv", {}).get("enabled", False)):
        train_df, val_df, _ = _split_kfold(cfg, df)
    else:
        train_df, val_df, _ = _split_single(cfg, df)
    # Same frames the experiment evaluates on (val_df is the binned test set)
    test_frames = build_isic_pc1_frames(cfg, train_df, val_df)

    classes = [str(c) for c in cfg.dataset.selected_classes]
    n_bins = len(test_frames)
    test_df = pd.concat(test_frames.values(), ignore_index=True)
    test_df["diagnosis"] = [classes[int(np.argmax(label))] for label in test_df["label"]]

    panels = {f"Test set (n={len(test_df)})": _counts_per_bin(test_df, classes, n_bins)}

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 4.2), squeeze=False)
    axes = axes[0]
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
        ax.set_xlabel("PC1 bin" if pc1_cfg.bin_by == "pc1" else "Distance-to-PC1 bin (near → far)")
        ax.set_ylabel("Number of images")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, bottom.max() * 1.1)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Diagnosis", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"isic_pc1: diagnoses per bin (bin_by={pc1_cfg.bin_by}, binning={pc1_cfg.binning})")
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
