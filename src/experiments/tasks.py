"""Evaluation tasks for OOD subgroup experiments.

Contains:
  - run_ood_subgroup_task: age / disease-count / gender subgroup OOD evaluation
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.utils.visualization import plot_uncertainty_line_plot, roc_simple

# ── Shared constants ──────────────────────────────────────────────────────

_OOD_UNCERTAINTY_KEYS = ("total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty")


def _concat_tensor_key(uncertainty_dict: dict, key: str) -> np.ndarray:
    v = uncertainty_dict[key]
    arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    # collapse per-class dim → scalar score per sample
    return arr.mean(axis=-1) if arr.ndim == 2 else arr


# ── OOD subgroup task ─────────────────────────────────────────────────────

def run_ood_subgroup_task(
    cfg,
    method,
    eval_loaders: Dict[str, DataLoader],
    level_names: List[str],
    result_dir: Path,
    plot_dir: Path,
) -> Dict[str, object]:
    """OOD detection evaluation for subgroup-split experiments (by_age, by_disease_count).

    Runs inference on each subgroup loader, treats the first subgroup as ID and
    all others as OOD, then computes:
      - Per-subgroup classification metrics (AUROC per class for multilabel)
      - OOD-detection AUROC for each uncertainty type (total / aleatoric / epistemic)
      - Uncertainty histograms (ID vs each OOD group)
      - ROC curves
      - Mean ± variance line plot across groups (for exactly 3 groups)
      - Misclassification detection AUROC
    Results are saved to JSON; plots are saved under *plot_dir*.
    """
    performance: dict = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # inference per subgroup 
    group_uncertainties: Dict[str, dict] = {}
    for name in level_names:
        cache = result_dir / f"ood_subgroup_{name}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                group_uncertainties[name] = pickle.load(f)
            print(f"  Loaded cached uncertainty for subgroup '{name}'")
        else:
            group_uncertainties[name] = method.measure_uncertainty(eval_loaders[name])
            with open(cache, "wb") as f:
                pickle.dump(group_uncertainties[name], f)
            print(f"  Computed uncertainty for subgroup '{name}'")

    id_name = level_names[0]
    ood_names = level_names[1:]

    # ── 2. Per-subgroup classification AUROC ──────────────────────────
    for name in level_names:
        u = group_uncertainties[name]
        preds = u["predictions"]
        gt = u["ground_truth"]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        try:
            if preds.ndim == 2 and gt.ndim == 2:
                # multilabel: mean per-class AUROC
                aucs = []
                for c in range(preds.shape[1]):
                    if len(np.unique(gt[:, c])) > 1:
                        aucs.append(roc_auc_score(gt[:, c], preds[:, c]))
                if aucs:
                    performance[f"classification_{name}_auroc"] = float(np.mean(aucs))
            elif preds.ndim == 2 and gt.ndim == 1:
                # single-label multiclass: preds are class probabilities
                present = np.unique(gt)
                if len(present) > 1:
                    p = preds[:, present]
                    p = p / p.sum(axis=1, keepdims=True)
                    performance[f"classification_{name}_auroc"] = float(
                        roc_auc_score(gt, p, multi_class="ovr", labels=present)
                    )
            else:
                if len(np.unique(gt)) > 1:
                    performance[f"classification_{name}_auroc"] = float(roc_auc_score(gt, preds))
        except Exception as e:
            print(f"  Classification AUROC for '{name}' failed: {e}")

    # ── 3. Build concatenated arrays for OOD detection ─────────────────
    # label: 0 = ID, 1..N = OOD groups
    id_arrays = {k: _concat_tensor_key(group_uncertainties[id_name], k) for k in _OOD_UNCERTAINTY_KEYS}
    ood_arrays_list = [
        {k: _concat_tensor_key(group_uncertainties[n], k) for k in _OOD_UNCERTAINTY_KEYS}
        for n in ood_names
    ]

    # ── 4. Uncertainty histograms ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, ut in enumerate(_OOD_UNCERTAINTY_KEYS):
        axes[idx].hist(id_arrays[ut], label=f"ID ({id_name})", bins=50, alpha=0.5)
        for i, oname in enumerate(ood_names):
            axes[idx].hist(ood_arrays_list[i][ut], label=f"OOD {i+1} ({oname})", bins=50, alpha=0.5)
        axes[idx].set_xlabel("Score")
        axes[idx].set_ylabel("Count")
        axes[idx].legend()
        axes[idx].set_title(ut)
        m_id = float(np.mean(id_arrays[ut]))
        m_oods = [float(np.mean(ood_arrays_list[i][ut])) for i in range(len(ood_names))]
        std_id = float(np.std(id_arrays[ut]))
        std_oods = [float(np.std(ood_arrays_list[i][ut])) for i in range(len(ood_names))]
        performance[f"ood_dist_{ut}"] = [m_id] + m_oods
        performance[f"ood_dist_{ut}_std"] = [std_id] + std_oods
    plt.tight_layout()
    fig.savefig(plot_dir / "ood_uncertainty_distributions.png")
    plt.close(fig)

    # ── 5. OOD detection AUROC ─────────────────────────────────────────
    # ID vs all OOD groups combined
    all_ood_arrays = {
        ut: np.concatenate([ood_arrays_list[i][ut] for i in range(len(ood_names))])
        for ut in _OOD_UNCERTAINTY_KEYS
    }
    for ut in _OOD_UNCERTAINTY_KEYS:
        id_s = id_arrays[ut]
        ood_s = all_ood_arrays[ut]
        if len(np.unique(np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))]))) > 1:
            try:
                auroc = float(roc_auc_score(
                    np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))]),
                    np.concatenate([id_s, ood_s]),
                ))
            except Exception:
                auroc = float("nan")
        else:
            auroc = float("nan")
        performance[f"ood_auroc_{ut}"] = auroc
        print(f"  OOD AUROC ({ut}): {auroc:.4f}")

    # ── 6. ROC curve (epistemic, ID vs all OOD) ────────────────────────
    roc_fig = roc_simple(
        id_scores=id_arrays["epistemic_uncertainty"],
        ood_scores=all_ood_arrays["epistemic_uncertainty"],
        plot_title="ROC Curve for OOD Detection",
    )
    roc_fig.savefig(plot_dir / "ood_roc.png")
    plt.close(roc_fig)

    # ── 7. Line plot across all groups ────────────────────────────────
    def _wrap(arrays_dict):
        return [{k: torch.tensor(arrays_dict[k]) for k in _OOD_UNCERTAINTY_KEYS}]

    groups = [(id_name, _wrap(id_arrays))]
    for name, ood_arrays in zip(ood_names, ood_arrays_list):
        groups.append((name, _wrap(ood_arrays)))

    line_fig = plot_uncertainty_line_plot(
        groups=groups,
        plot_title="Uncertainty Across Subgroups",
    )
    line_fig.savefig(plot_dir / "ood_line_plot.png")
    plt.close(line_fig)

    # ── 8. Misclassification detection (ID group only) ─────────────────
    u_id = group_uncertainties[id_name]
    preds_id = u_id["predictions"]
    gt_id = u_id["ground_truth"]
    if isinstance(preds_id, torch.Tensor):
        preds_id = preds_id.detach().cpu().numpy()
    if isinstance(gt_id, torch.Tensor):
        gt_id = gt_id.detach().cpu().numpy()
    if gt_id is not None and preds_id is not None:
        if preds_id.ndim == 2 and gt_id.ndim == 2:
            # multilabel
            miscls = ((preds_id > 0.5) != gt_id.astype(bool)).any(axis=-1)
        elif preds_id.ndim == 2 and gt_id.ndim == 1:
            # single-label multiclass: compare predicted class index with gt
            miscls = (preds_id.argmax(axis=-1) != gt_id)
        else:
            miscls = ((preds_id > 0.5) != gt_id.astype(bool))
        miscls = miscls.astype(int)
        for ut in _OOD_UNCERTAINTY_KEYS:
            try:
                auroc = float(roc_auc_score(miscls, id_arrays[ut])) if len(np.unique(miscls)) > 1 else float("nan")
            except Exception:
                auroc = float("nan")
            performance[f"miscls_auroc_{ut}"] = auroc
            print(f"  Misclassification AUROC ({ut}): {auroc:.4f}")

    with open(result_dir / "ood_subgroup_performance.json", "w") as f:
        json.dump(performance, f, indent=4)
    print(f"  Saved OOD subgroup results → {result_dir / 'ood_subgroup_performance.json'}")
    return performance
