import io, json, os, pickle
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
    'isic_age':         ['under_30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85'],
    'isic_skin_tone':   ['tone_1', 'tone_2', 'tone_3', 'tone_4', 'tone_5'],
    'isic_hair':        ['level_1', 'level_2', 'level_3', 'level_4'],
    'isic_drop':        ['level_1', 'level_2', 'level_3'],
    'isic_ink':         ['level_1', 'level_2', 'level_3', 'level_4'],

    # MNIST 
    'mnist_blur':       ['plain', 'low_severity', 'mid_severity', 'high_severity'],
    'mnist_fracture':   ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'mnist_thinning':   ['0.1', '0.3', '0.5', '0.7', '0.9'],

    # NIH
    'nih_age':          ['age_0', 'age_10', 'age_20', 'age_30', 'age_40', 'age_50', 'age_60', 'age_70', 'age_80', 'age_90'],
    'nih_disease':      [f'disease_{i}' for i in range(7)],
    'nih_gender':       ['Male', 'Female'],

    # CHEXPERT
    'chexpert_age':     ['age_0', 'age_10', 'age_20', 'age_30', 'age_40', 'age_50', 'age_60', 'age_70', 'age_80', 'age_90'],
    'chexpert_disease': [f'disease_{i}' for i in range(7)],
    'chexpert_gender':  ['Male', 'Female'],


    # VIN
    'vin_disease':      [f'disease_{i}' for i in range(7)],
}
NO_FINDINGS = {
    'isic' : 'isic_bin_plot(EXPERIMENT)',


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

    # NIH 
    'nih_age':         'Age',
    'nih_disease':     'Disease count', 
    'nih_gender':      'Gender',

    # CHEXPERT
    'chexpert_age':     'Age group',
    'chexpert_disease': 'Disease count',
    'chexpert_gender':  'Gender',

    # VIN
    'vin_disease':       'Disease count'       

}

root = Path(__file__).parent.parent / 'results'

results = {}

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
def load_results(date_overrides=None, experiments=None):
    date_overrides = date_overrides or {}
    results.clear()
    for exp in (experiments if experiments is not None else list(ORDERS)):
        date = date_overrides.get(exp)
        results[exp] = root / date / exp

    return results



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
        u_types = ['total_uncertainty','aleatoric_uncertainty', 'epistemic_uncertainty', ]
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
def _method_dir(rd, m):
    """Resolve a method's result dir case-insensitively (e.g. 'TTA' vs 'tta' on disk)."""
    for cand in (m, m.lower(), m.upper()):
        if (rd / cand).is_dir():
            return rd / cand
    return None


def _amb_task_auroc_barplot(experiment, json_key, title, filename_prefix, save=True):
    """Shared helper: bar chart of a single AUROC field from amb_task_performance.json per method."""
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd = results[experiment]

    rows = []
    for m in methods['total_uncertainty']:
        mdir = _method_dir(rd, m)
        if mdir is None:
            continue
        perf_path = mdir / 'amb_task_performance.json'
        if not perf_path.exists():
            continue
        with open(perf_path) as f:
            perf = json.load(f)
        auroc = perf.get(json_key, float('nan'))
        rows.append({'Method': legend_map.get(m, m), 'AUROC': auroc})

    if not rows:
        print(f'No amb_task_performance.json results found for {experiment}'); return

    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Method', y='AUROC', data=res, palette=palette_dict, ax=ax)
    ymin = min(0.4, max(0.0, res['AUROC'].min() - 0.05))
    ax.set_ylim(ymin, 1.0)
    ax.set_ylabel('AUROC')
    ax.set_title(f'{experiment}  —  {title}')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    if save:
        _save(fig, f'{filename_prefix}_{experiment}.pdf')
    plt.show()


def plot_misclassification_auroc(experiment, save=True):
    """Bar chart — AUROC of total uncertainty for detecting misclassified predictions
    among non-ambiguous (clear) samples, per method."""
    _amb_task_auroc_barplot(
        experiment,
        json_key='miscls_auroc_total_uncertainty',
        title='Misclassification detection on clear samples (total uncertainty)',
        filename_prefix='misclassification_auroc',
        save=save,
    )


def plot_ambiguity_auroc(experiment, save=True):
    """Bar chart — AUROC of total uncertainty for detecting ambiguous vs clear samples, per method."""
    _amb_task_auroc_barplot(
        experiment,
        json_key='amb_auroc_total_uncertainty',
        title='Ambiguity detection (total uncertainty)',
        filename_prefix='ambiguity_auroc',
        save=save,
    )


def _amb_task_auroc_compare_barplot(exp1, exp2, json_key, title, filename_prefix, save=True):
    """Shared helper: grouped bar chart comparing an amb_task AUROC field across two experiments."""
    rows = []
    for exp in (exp1, exp2):
        if exp not in results:
            print(f'{exp} not available'); continue
        rd = results[exp]
        for m in methods['total_uncertainty']:
            mdir = _method_dir(rd, m)
            if mdir is None:
                continue
            perf_path = mdir / 'amb_task_performance.json'
            if not perf_path.exists():
                continue
            with open(perf_path) as f:
                perf = json.load(f)
            auroc = perf.get(json_key, float('nan'))
            rows.append({'Method': legend_map.get(m, m), 'Experiment': exp, 'AUROC': auroc})

    if not rows:
        print(f'No amb_task_performance.json results found for {exp1} / {exp2}'); return

    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x='Method', y='AUROC', hue='Experiment', data=res, ax=ax)
    ymin = min(0.4, max(0.0, res['AUROC'].min() - 0.05))
    ax.set_ylim(ymin, 1.0)
    ax.set_ylabel('AUROC')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Dataset')
    plt.tight_layout()
    if save:
        _save(fig, f'{filename_prefix}_{exp1}_vs_{exp2}.pdf')
    plt.show()


def plot_misclassification_auroc_compare(exp1, exp2, save=True):
    """Grouped bar chart — misclassification-detection AUROC (total uncertainty), per method,
    comparing two experiments side by side (e.g. 'vin_amb' vs 'chexpert_amb')."""
    _amb_task_auroc_compare_barplot(
        exp1, exp2,
        json_key='miscls_auroc_aleatoric_uncertainty',
        title=f'{exp1} vs {exp2}  —  Misclassification detection on clear samples (Aleatoric uncertainty)',
        filename_prefix='misclassification_auroc_compare',
        save=save,
    )


def plot_ambiguity_auroc_compare(exp1, exp2, save=True):
    """Grouped bar chart — ambiguity-detection AUROC (total uncertainty), per method,
    comparing two experiments side by side (e.g. 'vin_amb' vs 'chexpert_amb')."""
    _amb_task_auroc_compare_barplot(
        exp1, exp2,
        json_key='amb_auroc_total_uncertainty',
        title=f'{exp1} vs {exp2}  —  Ambiguity detection (total uncertainty)',
        filename_prefix='ambiguity_auroc_compare',
        save=save,
    )


# ── Plot 5: All methods combined — 1×3 (one panel per uncertainty type) ───────
def plot_combined_methods(experiment, normalize=False, sharey=True, save=True):
    """1×3 figure: each panel shows all methods as coloured lines for one uncertainty type.

    normalize: min-max scale each panel to [0,1] so all three panels share the same y-axis.
    """
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd      = results[experiment]
    order   = ORDERS[experiment]
    xlabel  = XLABELS[experiment]
    u_types = ['epistemic_uncertainty', 'aleatoric_uncertainty', 'total_uncertainty']

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=sharey)
    fig.suptitle(f'{experiment}  —  all methods combined'
                 + ('  (min-max normalised per uncertainty type)' if normalize else ''), fontsize=14)

    for ax, u_type in zip(axes, u_types):
        m_list    = methods.get(u_type, methods['total_uncertainty'])
        available = [m for m in m_list if (rd / m).is_dir()]

        all_ys = {m: [_ood_mean_std(rd, m, lv, u_type)[0] for lv in order] for m in available}
        if normalize:
            all_vals = [v for ys in all_ys.values() for v in ys if not np.isnan(v)]
            vmin, vmax = min(all_vals), max(all_vals)
            denom = (vmax - vmin) or 1
            all_ys = {m: [(v - vmin) / denom for v in ys] for m, ys in all_ys.items()}

        for method, ys in all_ys.items():
            label = legend_map.get(method, method)
            ax.plot(order, ys, label=label, color=palette_dict.get(label),
                    linewidth=1.5, marker='o', markersize=3)
        ax.set_title(u_type.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Normalised score' if normalize else 'Uncertainty Score', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save:
        _save(fig, f'combined_methods{"_norm" if normalize else ""}_{experiment}.pdf')
    plt.show()


# selected_classes order as used in training (ground_truth integers map to this list)
_ISIC_SELECTED_CLASSES = ["BCC", "SCCKA", "AKIEC", "NV", "BKL", "MEL"]
_NV_IDX = _ISIC_SELECTED_CLASSES.index("NV")  # 3

def isic_bin_plot(experiment, focus_label=None, save=True):
    """Grouped bar plot: total samples vs sick samples (label != NV) per age bin.

    focus_label: class name string to highlight instead of all non-NV classes.
    """
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd        = results[experiment]
    order     = ORDERS[experiment]
    xlabel    = XLABELS[experiment]
    available = [m for m in methods['total_uncertainty'] if (rd / m).is_dir()]
    ref       = available[0] if available else None

    if ref is None:
        print('No method results found'); return

    # resolve focus_label to a class index if given
    if focus_label is not None and focus_label in _ISIC_SELECTED_CLASSES:
        sick_idx = _ISIC_SELECTED_CLASSES.index(focus_label)
        sick_label = focus_label
    else:
        sick_idx = None  # any non-NV
        sick_label = 'Sick (not NV)'

    totals, sicks = [], []
    for lv in order:
        p = rd / ref / f'ood_uncertainty_{lv}.pkl'
        d  = load_pkl(p)
        gt = d.get('ground_truth')
        if gt is None:
            totals.append(0); sicks.append(0); continue
        gt = _to_np(gt).astype(int)
        totals.append(len(gt))
        if sick_idx is not None:
            sicks.append(int((gt == sick_idx).sum()))
        else:
            sicks.append(int((gt != _NV_IDX).sum()))

    x      = np.arange(len(order))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 5))
    ax.bar(x - width / 2, totals, width, label='Total',      color='steelblue',  alpha=0.85)
    ax.bar(x + width / 2, sicks,  width, label=sick_label,   color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Sample count')
    ax.set_title(f'{experiment}  —  Sample distribution by {xlabel.lower()}')
    ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, f'isic_bin_{experiment}.pdf')
    plt.show()


def _nih_bin_counts(csv_path: Path, order: list, experiment: str) -> dict:
    """Count total and sick (not 'No Finding') samples per bin from NIH CSV."""
    df = pd.read_csv(csv_path)
    df["_age"] = (
        df["Patient Age"].astype(str).str.rstrip("Yy").str.strip()
        .pipe(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )
    df["_no_finding"] = df["Finding Labels"].astype(str).str.strip() == "No Finding"

    if "age" in experiment:
        df["_bin"] = (df["_age"] // 10 * 10).apply(lambda x: f"age_{x}")
    elif "gender" in experiment:
        df["_bin"] = df["Patient Gender"].astype(str).str.strip()
    elif "disease" in experiment:
        def _count(s):
            parts = [p for p in str(s).split("|") if p.strip() and p.strip() != "No Finding"]
            return len(parts)
        df["_bin"] = df["Finding Labels"].apply(_count).apply(lambda c: f"disease_{c}")
    else:
        return {}

    totals    = df.groupby("_bin").size().to_dict()
    no_find   = df[df["_no_finding"]].groupby("_bin").size().to_dict()
    return {k: (totals.get(k, 0), totals.get(k, 0) - no_find.get(k, 0)) for k in order}


def _chex_bin_counts(csv_path: Path, order: list, experiment: str) -> dict:
    """Count total and sick (No Finding != 1) samples per bin from CheXpert CSV."""
    df = pd.read_csv(csv_path)
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"]
    df["_no_finding"] = df["No Finding"].fillna(0).astype(float) == 1.0

    if "age" in experiment:
        df["_bin"] = (pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int) // 10 * 10).apply(
            lambda x: f"age_{x}")
    elif "gender" in experiment or "sex" in experiment:
        df["_bin"] = df["Sex"].astype(str).str.strip()
    else:
        return {}

    totals  = df.groupby("_bin").size().to_dict()
    no_find = df[df["_no_finding"]].groupby("_bin").size().to_dict()
    return {k: (totals.get(k, 0), totals.get(k, 0) - no_find.get(k, 0)) for k in order}


def _pkl_bin_counts(rd: Path, ref: str, order: list) -> dict:
    """Fallback: count total and sick (row_sum > 0) from pkl ground_truth (VinDr)."""
    out = {}
    for lv in order:
        p = rd / ref / f'ood_uncertainty_{lv}.pkl'
        if not p.exists():
            out[lv] = (0, 0); continue
        d  = load_pkl(p)
        gt = d.get('ground_truth')
        if gt is None:
            out[lv] = (0, 0); continue
        gt = np.asarray(gt.detach().cpu().numpy() if hasattr(gt, 'detach') else gt)
        n  = gt.shape[0]
        sick = int((gt.sum(axis=1) > 0).sum()) if gt.ndim > 1 else int((gt > 0).sum())
        out[lv] = (n, sick)
    return out


def chest_bin_plot(experiment, data_root=None, save=True):
    """Grouped bar plot: total samples vs sick samples per subgroup bin.

    NIH:      loads Data_Entry_2017.csv; sick = Finding Labels is not 'No Finding'.
    CheXpert: loads valid.csv; sick = 'No Finding' column != 1.
    VinDr:    reads pkl ground_truth; sick = any positive label (row_sum > 0).
    """
    if experiment not in results:
        print(f'{experiment} not available'); return
    rd     = results[experiment]
    order  = ORDERS[experiment]
    xlabel = XLABELS[experiment]

    if data_root is None:
        # root = …/results, so parent is the workspace root
        data_root = root.parent / 'data'
    data_root = Path(data_root)

    exp = experiment.lower()
    counts: dict = {}

    if exp.startswith("nih"):
        csv_path = data_root / 'NIH' / 'Data_Entry_2017.csv'
        if csv_path.exists():
            counts = _nih_bin_counts(csv_path, order, exp)
        else:
            print(f'NIH CSV not found at {csv_path}; falling back to pkl')

    elif exp.startswith("chex"):
        csv_path = data_root / 'chexpert' / 'valid.csv'
        if csv_path.exists():
            counts = _chex_bin_counts(csv_path, order, exp)
        else:
            print(f'CheXpert CSV not found at {csv_path}; falling back to pkl')

    if not counts:
        # VinDr or CSV not available: fall back to pkl
        available = [m for m in methods['total_uncertainty'] if (rd / m).is_dir()]
        ref = available[0] if available else None
        if ref is None:
            print('No method results found'); return
        counts = _pkl_bin_counts(rd, ref, order)

    totals = [counts.get(lv, (0, 0))[0] for lv in order]
    sicks  = [counts.get(lv, (0, 0))[1] for lv in order]

    x     = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 5))
    ax.bar(x - width / 2, totals, width, label='Total', color='steelblue',  alpha=0.85)
    ax.bar(x + width / 2, sicks,  width, label='Sick',  color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Sample count')
    ax.set_title(f'{experiment}  —  Sample distribution by {xlabel.lower()}')
    ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, f'chest_bin_{experiment}.pdf')
    plt.show()
