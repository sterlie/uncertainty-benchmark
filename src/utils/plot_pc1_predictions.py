"""Ground-truth vs predicted diagnoses per PC1 test bin for the isic_pc1 task.

Reads the cached per-bin outputs (ood_uncertainty_bin_*.pkl) written by run_experiment and draws,
for each method, two stacked bars per bin: ground truth (left) and predicted label (right).

Usage (from repo root):
    python -m src.utils.plot_pc1_predictions results/2026-10-29/isic_pc1
    python -m src.utils.plot_pc1_predictions results/2026-10-29/isic_pc1 --run run_0
"""

import argparse
import math
import os
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.datasets.isic import DEFAULT_SELECTED_CLASSES
from src.utils.plot_pc1_bins import COLORS

METHOD_NAMES = {
    "mc_dropout": "MC Dropout",
    "ensemble": "Deep Ensemble",
    "swag": "SWAG",
    "laplace_approximation": "Laplace Approximation",
    "ddu": "DDU",
    "tta": "Test-Time Augmentation",
    "het_xl": "HET-XL",
    "entropy": "Entropy",
}


def _to_numpy(v) -> np.ndarray:
    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


def _load_method(method_dir: Path):
    """Return {bin_name: (gt, pred)} with integer class indices, bins ordered by number."""
    files = sorted(method_dir.glob("ood_uncertainty_bin_*.pkl"),
                   key=lambda p: int(re.search(r"bin_(\d+)", p.name).group(1)))
    out = {}
    for path in files:
        with open(path, "rb") as f:
            u = pickle.load(f)
        gt = _to_numpy(u["ground_truth"])
        gt = gt.argmax(axis=-1) if gt.ndim == 2 else gt
        pred = _to_numpy(u["predicted_labels"]) if "predicted_labels" in u else _to_numpy(u["predictions"]).argmax(axis=-1)
        out[path.stem.replace("ood_uncertainty_", "")] = (gt.astype(int), pred.astype(int))
    return out


def _plot_method(ax, per_bin, classes, title):
    bin_names = list(per_bin.keys())
    x = np.arange(len(bin_names))
    width = 0.38
    for offset, which in ((-width / 2 - 0.01, 0), (width / 2 + 0.01, 1)):
        bottom = np.zeros(len(bin_names))
        for c, color in enumerate(COLORS[:len(classes)]):
            counts = np.array([(per_bin[b][which] == c).sum() for b in bin_names])
            ax.bar(x + offset, counts, bottom=bottom, width=width, color=color,
                   edgecolor="white", linewidth=1.2, hatch=None if which == 0 else "//")
            bottom += counts
        for xi in x:
            ax.text(xi + offset, -0.5, "GT" if which == 0 else "Pred", ha="center", va="top",
                    fontsize=7, color="#52514e")

    for xi, b in zip(x, bin_names):
        gt, pred = per_bin[b]
        acc = float((gt == pred).mean()) if len(gt) else float("nan")
        total = max(len(gt), len(pred))
        ax.text(xi, total + 0.5, f"acc {acc:.2f}", ha="center", va="bottom", fontsize=8, color="#52514e")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_names)
    ax.tick_params(axis="x", pad=12)
    ax.set_ylabel("Number of images")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(len(v[0]) for v in per_bin.values()) * 1.15)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", help="results/<date>/isic_pc1 directory")
    parser.add_argument("--run", default="", help="run sub-directory (e.g. run_0) when num_runs > 1")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_SELECTED_CLASSES,
                        help="class names in label order (dataset.selected_classes)")
    parser.add_argument("--out", default=os.path.join("src", "plots", "pc1_gt_vs_pred_per_bin"),
                        help="output path without extension")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    methods = {}
    for method_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        per_bin = _load_method(method_dir / args.run)
        if per_bin:
            methods[method_dir.name] = per_bin
    if not methods:
        raise SystemExit(f"No ood_uncertainty_bin_*.pkl files found under {exp_dir}")

    ncols = min(4, len(methods))
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (name, per_bin) in zip(axes.flat, methods.items()):
        _plot_method(ax, per_bin, args.classes, METHOD_NAMES.get(name, name))
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)

    handles = [Patch(facecolor=c, label=cls) for cls, c in zip(args.classes, COLORS)]
    handles += [Patch(facecolor="white", edgecolor="#52514e", label="Ground truth"),
                Patch(facecolor="white", edgecolor="#52514e", hatch="//", label="Predicted")]
    fig.legend(handles=handles, title="Diagnosis", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{exp_dir.name}: ground truth vs predicted diagnosis per PC1 test bin")
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=150)
    print({"plot": f"{args.out}.png"})

    for name, per_bin in methods.items():
        print(f"\n{name}")
        for b, (gt, pred) in per_bin.items():
            gt_c = np.bincount(gt, minlength=len(args.classes))
            pred_c = np.bincount(pred, minlength=len(args.classes))
            print(f"  {b}: acc={float((gt == pred).mean()):.2f}  "
                  f"gt={dict(zip(args.classes, gt_c.tolist()))}  pred={dict(zip(args.classes, pred_c.tolist()))}")


if __name__ == "__main__":
    main()
