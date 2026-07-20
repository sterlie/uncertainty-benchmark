"""Evaluation task: predict uncertainty type (aleatoric vs epistemic) on MNIST.

Each image in the combined eval set is either:
  - Blurred  → ground-truth label: aleatoric  (model was trained with blur)
  - Fractured → ground-truth label: epistemic  (never seen during training)

The predictor is the epistemic fraction:
    score = epistemic / (epistemic + aleatoric + eps)

AUROC of this score classifying fracture (epistemic) vs blur (aleatoric) is the
main metric — 1.0 means perfect uncertainty-type attribution.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


_EPS = 1e-10
_UQ_KEYS = ("total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty")


def _to_numpy(v) -> np.ndarray:
    arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    return arr.mean(axis=-1) if arr.ndim == 2 else arr


def run_uncertainty_decomposition_(
    cfg,
    method,
    eval_loaders: Dict[str, DataLoader],
    level_names: List[str],
    result_dir: Path,
    plot_dir: Path,
    expected_uq_type
):
    
    performance: dict = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Inference per loader (cached) ─────────────────────────────────
    uncertainties: Dict[str, dict] = {}
    for name in level_names:
        cache = result_dir / f"uncertainty_decomp_{name}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                uncertainties[name] = pickle.load(f)
            print(f"  Loaded cached uncertainty for '{name}'")
        else:
            uncertainties[name] = method.measure_uncertainty(eval_loaders[name])
            with open(cache, "wb") as f:
                pickle.dump(uncertainties[name], f)
            print(f"  Computed uncertainty for '{name}'")

    arrays: Dict[str, Dict[str, np.ndarray]] = {
        name: {k: _to_numpy(uncertainties[name][k]) for k in _UQ_KEYS}
        for name in level_names
    }

    # Determine the "other" uncertainty type
    other_uq = "epistemic_uncertainty" if expected_uq_type == "aleatoric_uncertainty" else "aleatoric_uncertainty"
    color = "steelblue" if expected_uq_type == "aleatoric_uncertainty" else "tomato"
    gt_label = "aleatoric" if expected_uq_type == "aleatoric_uncertainty" else "epistemic"

    # ── 3. Per-level AUROC ────────────────────────────────────────────────
    # true label = 1 if uncertainty_gt dominates (uncertainty_gt > other_uq), else 0
    # score      = uncertainty_gt value itself
    level_aurocs: Dict[str, float] = {}
    for name in level_names:
        gt_scores    = arrays[name][expected_uq_type]
        other_scores = arrays[name][other_uq]
        true_labels  = (gt_scores > other_scores).astype(int)
        n_pos = int(true_labels.sum())
        n_neg = len(true_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            auroc = float("nan")
            print(f"  [{name}] AUROC=nan  (all labels identical: {n_pos} pos, {n_neg} neg)")
        else:
            auroc = float(roc_auc_score(true_labels, gt_scores))
            print(f"  [{name}] AUROC={auroc:.4f}  ({n_pos} pos / {n_neg} neg)")
        level_aurocs[name] = auroc
        performance[f"auroc_{expected_uq_type}_{name}"] = auroc

    # ── 4. Per-level means ────────────────────────────────────────────────
    for name in level_names:
        for k in _UQ_KEYS:
            performance[f"mean_{k}_{name}"] = float(np.mean(arrays[name][k]))

    # ── 5. AUROC-per-level line plot ──────────────────────────────────────
    x = np.arange(len(level_names))
    valid_aurocs = [v for v in level_aurocs.values() if not np.isnan(v)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, [level_aurocs[n] for n in level_names], marker="o", color=color)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(level_names, rotation=30, ha="right")
    ax.set_ylabel(f"AUROC  ({gt_label} dominant vs not)")
    ax.set_title(f"Per-level AUROC — {gt_label} eval set\n(score={expected_uq_type}, label=1 if dominant)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "uncertainty_type_auroc_per_level.png", bbox_inches="tight")
    plt.close(fig)

    # ── 6. Bar plot: mean uncertainties per level ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ukey, title in zip(
        axes,
        ("aleatoric_uncertainty", "epistemic_uncertainty"),
        ("Aleatoric uncertainty", "Epistemic uncertainty"),
    ):
        means = [np.mean(arrays[n][ukey]) for n in level_names]
        stds  = [np.std(arrays[n][ukey])  for n in level_names]
        ax.bar(x, means, yerr=stds, color=color, capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(level_names, rotation=30, ha="right")
        ax.set_title(title + (" (↑ expected)" if ukey == expected_uq_type else " (↓ expected)"))
        ax.set_ylabel("Mean uncertainty")
    plt.tight_layout()
    fig.savefig(plot_dir / "uncertainty_type_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ── 7. Save results ───────────────────────────────────────────────────
    with open(result_dir / "uncertainty_decomposition.json", "w") as f:
        json.dump(performance, f, indent=4)
    print(f"  Saved → {result_dir / 'uncertainty_decomposition.json'}")
    return performance



def run_uncertainty_decomposition(
    cfg,
    method,
    eval_loaders: Dict[str, DataLoader],
    level_names: List[str],
    result_dir: Path,
    plot_dir: Path,
) -> Dict[str, object]:
    """Predict uncertainty type (aleatoric vs epistemic) for a mixed eval set.

    Args:
        eval_loaders: dict mapping name → DataLoader.
                      Names containing "blur" are treated as aleatoric OOD.
                      Names containing "fracture" are treated as epistemic OOD.
        level_names:  ordered list of keys in eval_loaders.

    Returns:
        performance dict with AUROC and mean uncertainty scores.
    """
    performance: dict = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Inference per loader (cached) ─────────────────────────────────
    uncertainties: Dict[str, dict] = {}
    for name in level_names:
        cache = result_dir / f"uncertainty_decomp_{name}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                uncertainties[name] = pickle.load(f)
            print(f"  Loaded cached uncertainty for '{name}'")
        else:
            uncertainties[name] = method.measure_uncertainty(eval_loaders[name])
            with open(cache, "wb") as f:
                pickle.dump(uncertainties[name], f)
            print(f"  Computed uncertainty for '{name}'")

    arrays: Dict[str, Dict[str, np.ndarray]] = {
        name: {k: _to_numpy(uncertainties[name][k]) for k in _UQ_KEYS}
        for name in level_names
    }

    # ── 2. Split into blur (aleatoric) vs fracture (epistemic) groups ────
    blur_names     = [n for n in level_names if "blur"     in n.lower()]
    fracture_names = [n for n in level_names if "fracture" in n.lower()]

    aleatoric_scores_blur  = np.concatenate([arrays[n]["aleatoric_uncertainty"] for n in blur_names])
    epistemic_scores_blur      = np.concatenate([arrays[n]["epistemic_uncertainty"] for n in blur_names])
    epistemic_score_fracture  = np.concatenate([arrays[n]["epistemic_uncertainty"] for n in fracture_names])
    aleatoric_scores_fracture      = np.concatenate([arrays[n]["aleatoric_uncertainty"] for n in fracture_names])

    # ── 3. Combined arrays with ground-truth labels ───────────────────────
    # label 0 = aleatoric (blur), label 1 = epistemic (fracture)
    all_aleatoric = np.concatenate([aleatoric_scores_blur,  aleatoric_scores_fracture])
    all_epistemic = np.concatenate([epistemic_scores_blur,      epistemic_score_fracture])
    all_labels    = np.concatenate([np.zeros(len(aleatoric_scores_blur)), np.ones(len(epistemic_score_fracture))])

    # normalize before computing score fraction 
    norm_aleatoric = (all_aleatoric - np.mean(all_aleatoric)) / np.std(all_aleatoric)    
    norm_epistemic = (all_epistemic - np.mean(all_epistemic)) / np.std(all_epistemic)    

    
    # Epistemic fraction score: high → predict epistemic
    ep_fraction = norm_epistemic / (norm_epistemic + norm_aleatoric + _EPS)

    # ── 4. Main metric: AUROC of epistemic fraction ───────────────────────
    auroc_fraction = float(roc_auc_score(all_labels, ep_fraction))
    performance["uncertainty_type_auroc_ep_fraction"] = auroc_fraction
    print(f"  Uncertainty-type AUROC (ep fraction): {auroc_fraction:.4f}  (1.0 = perfect)")

    # Also report raw aleatoric and epistemic scores individually
    for ukey, scores in [("aleatoric_uncertainty", all_aleatoric), ("epistemic_uncertainty", all_epistemic)]:
        auroc = float(roc_auc_score(all_labels, scores))
        performance[f"uncertainty_type_auroc_{ukey}"] = auroc
        print(f"  Uncertainty-type AUROC ({ukey}): {auroc:.4f}")

    # Per-group means
    for name in level_names:
        for k in _UQ_KEYS:
            performance[f"mean_{k}_{name}"] = float(np.mean(arrays[name][k]))

    # ── 5. Scatter: aleatoric vs epistemic per image ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(aleatoric_scores_blur, epistemic_scores_blur,  alpha=0.3, s=8, color="steelblue", label=f"blur (aleatoric, n={len(aleatoric_scores_blur)})")
    ax.scatter(aleatoric_scores_fracture,     epistemic_score_fracture, alpha=0.3, s=8, color="tomato",    label=f"fracture (epistemic, n={len(epistemic_score_fracture)})")
    ax.set_xlabel("Aleatoric uncertainty")
    ax.set_ylabel("Epistemic uncertainty")
    ax.set_title(f"Uncertainty decomposition\n(ep-fraction AUROC = {auroc_fraction:.3f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "uncertainty_type_scatter.png")
    plt.close(fig)

    # ── 6. Histogram of epistemic fraction per group ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ep_fraction[:len(aleatoric_scores_blur)],  bins=40, alpha=0.6, color="steelblue", label="blur (aleatoric)")
    ax.hist(ep_fraction[len(aleatoric_scores_blur):],  bins=40, alpha=0.6, color="tomato",    label="fracture (epistemic)")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Epistemic fraction  [epistemic / (epistemic + aleatoric)]")
    ax.set_ylabel("Count")
    ax.set_title("Predicted uncertainty type distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "uncertainty_type_histogram.png")
    plt.close(fig)

    # ── 7. Bar plot: mean aleatoric & epistemic per severity level ────────
    all_names = blur_names + fracture_names
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(all_names))
    for ax, ukey, title in zip(
        axes,
        ("aleatoric_uncertainty", "epistemic_uncertainty"),
        ("Aleatoric (↑ expected for blur)", "Epistemic (↑ expected for fracture)"),
    ):
        means  = [np.mean(arrays[n][ukey]) for n in all_names]
        stds   = [np.std(arrays[n][ukey])  for n in all_names]
        colors = ["steelblue" if n in blur_names else "tomato" for n in all_names]
        ax.bar(x, means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Mean uncertainty")
    fig.legend(
        handles=[Patch(color="steelblue", label="blur (aleatoric)"), Patch(color="tomato", label="fracture (epistemic)")],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "uncertainty_type_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ── 8. Save results ───────────────────────────────────────────────────
    with open(result_dir / "uncertainty_decomposition.json", "w") as f:
        json.dump(performance, f, indent=4)
    print(f"  Saved → {result_dir / 'uncertainty_decomposition.json'}")
    return performance


