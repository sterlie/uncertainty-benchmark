import io, os, pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score


class _CPUUnpickler(pickle.Unpickler):
    """Unpickler that remaps CUDA tensors to CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load_pkl(path):
    with open(path, 'rb') as f:
        return _CPUUnpickler(f).load()


# ── OOD levels per subgroup ───────────────────────────────────────────────────
ORDERS = {
    # ISIC
    'isic_age':       ['under_30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85'],
    'isic_skin_tone': ['tone_1', 'tone_2', 'tone_3', 'tone_4', 'tone_5'],
    'isic_hair':      ['level_1', 'level_2', 'level_3', 'level_4'],
    'isic_drop':      ['level_1', 'level_2', 'level_3'],
    'isic_ink':       ['level_1', 'level_2', 'level_3', 'level_4'],
    # MNIST
    'mnist_blur':     ['plain', 'low_severity', 'mid_severity', 'high_severity'],
    'mnist_fracture': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'mnist_thinning': ['0.1', '0.3', '0.5', '0.7', '0.9'],
}
XLABELS = {
    # ISIC
    'isic_age':       'Age',
    'isic_skin_tone': 'Skin Tone',
    'isic_hair':      'Hair',
    'isic_drop':      'Level of dermoscopy liquid',
    'isic_ink':       'Ink level',
    # MNIST
    'mnist_blur':     'Level of blurring',
    'mnist_fracture': 'Number of fractures',
    'mnist_thinning': 'Thinning strength',
}

root = Path(__file__).parent.parent / 'results'


def latest_result(experiment, date=None):
    """Return the most-recent result directory for an experiment."""
    available_dates = sorted(
        d.name for d in os.scandir(root)
        if d.is_dir() and (root / d.name / experiment).is_dir()
    )
    if not available_dates:
        raise FileNotFoundError(f'No results found for {experiment}')
    chosen = date if date else available_dates[-1]
    exp_root = root / chosen / experiment
    newest_time = max(
        (d for d in os.scandir(exp_root) if d.is_dir()),
        key=lambda d: d.stat().st_mtime
    ).name
    return exp_root / newest_time


results = {}


def load_results(date_overrides=None, experiments=None):
    """Discover and cache result directories for all known experiments.

    Parameters
    ----------
    date_overrides : dict, optional
        Mapping of experiment name → ``'YYYY-MM-DD'`` to pin a specific date.
        Omit a key (or pass ``None``) to use the most-recent available date.
    experiments : list, optional
        Subset of experiment names to load.  Defaults to all keys in ORDERS.
    """
    date_overrides = date_overrides or {}
    results.clear()
    for exp in (experiments if experiments is not None else list(ORDERS)):
        date = date_overrides.get(exp)
        try:
            results[exp] = latest_result(exp, date)
            print(f"  {exp}: {results[exp]}")
        except FileNotFoundError as e:
            print(f"  {exp}: not found — {e}")
    return results




legend_map = {
    'mc_dropout':            'MC Dropout',
    'ensemble':              'Deep Ensemble',
    'swag':                  'SWAG',
    'laplace_approximation': 'Laplace Approximation',
    'ddu':                   'DDU',
    'TTA':                   'Test-Time Augmentation',
    'het_xl':                'HET-XL',
    'entropy':               'Entropy',
}
methods = {
    'epistemic_uncertainty': ['mc_dropout', 'ensemble', 'swag', 'laplace_approximation', 'ddu', 'het_xl'],
    'aleatoric_uncertainty': ['mc_dropout', 'ensemble', 'swag', 'laplace_approximation', 'ddu', 'TTA', 'het_xl', 'entropy'],
    'total_uncertainty':     ['mc_dropout', 'ensemble', 'swag', 'laplace_approximation', 'ddu', 'TTA', 'het_xl', 'entropy'],
}
palette_dict = dict(zip(legend_map.values(), sns.color_palette('tab10', len(legend_map))))
U_COLORS = {
    'total_uncertainty':     'steelblue',
    'aleatoric_uncertainty': 'darkorange',
    'epistemic_uncertainty': 'forestgreen',
}


# ── Shared helpers ────────────────────────────────────────────────────────────
def _to_np(v):
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, 'numpy') else v).ravel()


def _ood_mean_std(rd, method, level, u_type):
    p = rd / method / f'ood_uncertainty_{level}.pkl'
    if not p.exists():
        return np.nan, 0, 0
    d = load_pkl(p)
    if u_type not in d:
        return np.nan, 0, 0
    arr = _to_np(d[u_type])
    return np.nanmean(arr), np.nanstd(arr), len(arr)


def _save(fig, name):
    out = Path(__file__).parent.parent / 'plots' / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved → {out}')


# ── Plot 1: Per-method OOD uncertainty grid (2 × 4) ──────────────────────────
def plot_method_grid(experiment, u_types=None, save=True):
    """2×4 grid of per-method OOD uncertainty line plots. Overlays all u_types on each panel."""
    if u_types is None:
        u_types = ['epistemic_uncertainty', 'aleatoric_uncertainty', 'total_uncertainty']
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd        = results[experiment]
    order     = ORDERS[experiment]
    xlabel    = XLABELS[experiment]
    available = [m for m in methods['total_uncertainty'] if (rd / m).is_dir()]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharey=False)
    fig.suptitle(f'{experiment}  —  uncertainty by method', fontsize=14)

    for i, method in enumerate(available[:8]):
        ax = axes[i // 4, i % 4]
        for u_type in u_types:
            clr = U_COLORS.get(u_type, 'steelblue')
            ys, errs = [], []
            for lv in order:
                m, s, _ = _ood_mean_std(rd, method, lv, u_type)
                ys.append(m); errs.append(s)
            ax.plot(order, ys, color=clr, linewidth=1.5, label=u_type.replace('_', ' '))
            ax.fill_between(order,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            alpha=0.15, color=clr)
        ax.set_title(legend_map.get(method, method), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('Uncertainty Score', fontsize=8)
        ax.tick_params(axis='x', rotation=60, labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(available), 8):
        axes[j // 4, j % 4].set_visible(False)

    plt.tight_layout()
    if save:
        _save(fig, f'method_grid_{experiment}.pdf')
    plt.show()


# ── Plot 2: Sample distribution per OOD level ────────────────────────────────
def plot_distribution(experiment, save=True):
    """Bar chart of sample counts for each OOD level."""
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd        = results[experiment]
    order     = ORDERS[experiment]
    xlabel    = XLABELS[experiment]
    available = [m for m in methods['total_uncertainty'] if (rd / m).is_dir()]
    ref       = available[0] if available else None

    counts = []
    for lv in order:
        p = rd / ref / f'ood_uncertainty_{lv}.pkl' if ref else None
        if p and p.exists():
            d = load_pkl(p)
            counts.append(len(_to_np(next(iter(d.values())))))
        else:
            counts.append(0)

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.8), 4))
    ax.bar(range(len(order)), counts, color='steelblue')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Sample count')
    ax.set_title(f'{experiment}  —  Distributions')
    plt.tight_layout()
    if save:
        _save(fig, f'distribution_{experiment}.pdf')
    plt.show()


# ── Plot 3: OOD-detection AUROC line plot ─────────────────────────────────────
def plot_ood_auroc(experiment, u_type='total_uncertainty', save=True):
    """Per-method OOD-detection AUROC line plot across OOD levels."""
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd        = results[experiment]
    order     = ORDERS[experiment]
    xlabel    = XLABELS[experiment]
    m_list    = methods.get(u_type, methods['total_uncertainty'])
    available = [m for m in m_list if (rd / m).is_dir()]

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.7), 5))
    fig.suptitle(f'{experiment}  —  OOD AUROC  ({u_type.replace("_", " ")})', fontsize=13)

    for method in available:
        id_path = rd / method / 'valid_uncertainties.pkl'
        if not id_path.exists():
            continue
        id_d = load_pkl(id_path)
        if u_type not in id_d:
            continue
        id_scores = _to_np(id_d[u_type])
        aurocs = []
        for lv in order:
            p = rd / method / f'ood_uncertainty_{lv}.pkl'
            if not p.exists():
                aurocs.append(np.nan); continue
            ood_d = load_pkl(p)
            if u_type not in ood_d:
                aurocs.append(np.nan); continue
            ood_scores = _to_np(ood_d[u_type])
            labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            try:
                aurocs.append(roc_auc_score(labels, scores))
            except Exception:
                aurocs.append(np.nan)
        label = legend_map.get(method, method)
        ax.plot(order, aurocs, label=label, color=palette_dict.get(label), linewidth=1.5, marker='o', markersize=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('AUROC')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Method', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    if save:
        _save(fig, f'ood_auroc_{experiment}_{u_type}.pdf')
    plt.show()


# ── Plot 4: Misclassification detection AUROC bar chart ──────────────────────
def plot_misclassification_auroc(experiment, save=True):
    """Bar chart — classification AUROC on the ID validation set per method."""
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd        = results[experiment]
    available = [m for m in methods['total_uncertainty'] if (rd / m).is_dir()]

    rows = []
    for m in available:
        id_path = rd / m / 'valid_uncertainties.pkl'
        if not id_path.exists():
            continue
        data  = load_pkl(id_path)
        gt    = _to_np(data['ground_truth'])
        preds = np.asarray(data['predictions'].detach().cpu().numpy()
                           if hasattr(data['predictions'], 'numpy') else data['predictions'])
        if preds.ndim == 3:
            preds = preds.mean(axis=0)
        n_classes = preds.shape[1] if preds.ndim > 1 else 2
        try:
            auroc = (roc_auc_score(gt, preds[:, 1]) if n_classes == 2
                     else roc_auc_score(gt, preds, multi_class='ovr'))
        except Exception as e:
            print(f'{m}: {e}'); auroc = float('nan')
        rows.append({'Method': legend_map.get(m, m), 'AUROC': auroc})

    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Method', y='AUROC', data=res, palette=palette_dict, ax=ax)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('AUROC')
    ax.set_title(f'{experiment}  —  Misclassification detection (total uncertainty)')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    if save:
        _save(fig, f'misclassification_auroc_{experiment}.pdf')
    plt.show()
