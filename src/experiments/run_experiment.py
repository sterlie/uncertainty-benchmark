import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.methods.method_factory import MethodFactory
from src.experiments.datasets import get_dataset_adapter
from src.experiments.tasks import run_ood_subgroup_task, run_uncertainty_decomposition

from src.utils.visualization import trend, entropy

# ── Ambiguity task ─────────────────────────────────────────────────────────

def _concat_uncertainty_key(results: list, key: str) -> np.ndarray:
    return np.concatenate([r[key].detach().cpu().numpy() for r in results])


def _plot_amb_distributions(results: list, targets_np: np.ndarray, plot_dir: Path, prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, ut in enumerate(["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"]):
        scores = _concat_uncertainty_key(results, ut)
        axes[idx].hist(scores[targets_np == 0], label="Clear", bins=50, alpha=0.5)
        axes[idx].hist(scores[targets_np == 1], label="Ambiguous", bins=50, alpha=0.5)
        axes[idx].set_xlabel("Score")
        axes[idx].set_ylabel("Count")
        axes[idx].legend()
        axes[idx].set_title(ut)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{prefix}_uncertainty_distributions.png")
    plt.close(fig)


def _eval_amb_detection(results: list, targets: torch.Tensor, performance: dict, prefix: str) -> dict:
    targets_np = targets.cpu().numpy()
    for ut in ["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"]:
        scores = _concat_uncertainty_key(results, ut)
        m_clear = float(np.mean(scores[targets_np == 0]))
        m_amb = float(np.mean(scores[targets_np == 1]))
        std_clear = float(np.std(scores[targets_np == 0]))
        std_amb = float(np.std(scores[targets_np == 1]))
        performance[f"{prefix}_dist_{ut}"] = [m_clear, m_amb]
        performance[f"{prefix}_dist_{ut}_std"] = [std_clear, std_amb]
        auroc = float(roc_auc_score(targets_np, scores)) if len(np.unique(targets_np)) > 1 else float("nan")
        performance[f"{prefix}_auroc_{ut}"] = auroc
        print(f"  {ut}: clear_mean={m_clear:.4f}  amb_mean={m_amb:.4f}  AUROC={auroc:.4f}")
    return performance


def run_ambiguous_uncertainty_task(
    cfg,
    method,
    val_loader: DataLoader,
    test_loader: DataLoader,
    result_dir: Path,
    plot_dir: Path,
    num_samples: int = -1,
) -> Dict[str, object]:
    """Tests whether uncertainty is higher for inherently ambiguous images.

    Enables negative_label=True on val/test datasets so ambiguous samples
    return label==-1, then evaluates AUROC for each uncertainty type.
    """
    performance: dict = {}

    val_ds = val_loader.dataset
    test_ds = test_loader.dataset
    val_ds.negative_label = True
    test_ds.negative_label = True

    if num_samples and num_samples > 0:
        val_ds = Subset(val_ds, range(min(num_samples, len(val_ds))))
        test_ds = Subset(test_ds, range(min(num_samples, len(test_ds))))

    combined_ds = ConcatDataset([val_ds, test_ds])
    combined_loader = DataLoader(
        combined_ds,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=getattr(test_loader, "num_workers", 0),
    )
    print(f"Ambiguity task: {len(val_ds)} val + {len(test_ds)} test = {len(combined_ds)} total")

    result_dir.mkdir(parents=True, exist_ok=True)
    cache_path = result_dir / "amb_task_results.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print("Loaded cached amb_task results.")
    else:
        results = [method.measure_uncertainty(combined_loader)]
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)

    all_predictions = torch.cat([r["predictions"] for r in results], dim=0)
    all_targets = torch.cat([r["ground_truth"] for r in results], dim=0)

    amb_targets = (all_targets == -1).any(dim=-1).long()
    n_amb = int(amb_targets.sum().item())
    n_clear = int((amb_targets == 0).sum().item())
    print(f"Ambiguous: {n_amb}  Clear: {n_clear}")

    if n_amb == 0:
        print(
            "WARNING: No ambiguous samples found (amb_targets all 0).  "
            "This usually means the dataset CSV has no -1 (uncertain) labels — "
            "e.g. CheXpert valid.csv uses only 0/1 labels.  "
            "To run the ambiguity task, point 'dataset.metadata_csv' to a CSV "
            "that contains -1 labels (e.g. the CheXpert train.csv).  "
            "Ambiguity-task metrics will be NaN."
        )
        for ut in ["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"]:
            performance[f"amb_dist_{ut}"] = [float("nan"), float("nan")]
            performance[f"amb_dist_{ut}_std"] = [float("nan"), float("nan")]
            performance[f"amb_auroc_{ut}"] = float("nan")
            print(f"  {ut}: clear_mean=nan  amb_mean=nan  AUROC=nan  (no ambiguous samples)")
    else:
        _plot_amb_distributions(results, amb_targets.cpu().numpy(), plot_dir, prefix="amb")
        performance = _eval_amb_detection(results, amb_targets, performance, prefix="amb")

    clear_mask = (amb_targets == 0)
    if clear_mask.sum() > 0:
        preds_clear = (all_predictions[clear_mask] > 0.5)
        true_clear = (all_targets[clear_mask] == 1)
        miscls_targets = (preds_clear != true_clear).any(dim=-1).long()
        miscls_results = [{
            k: v[clear_mask] if isinstance(v, torch.Tensor) and v.shape[0] == len(all_targets) else v
            for k, v in r.items()
        } for r in results]
        print(f"Misclassification: {miscls_targets.sum().item()} / {clear_mask.sum().item()}")
        _plot_amb_distributions(miscls_results, miscls_targets.cpu().numpy(), plot_dir, prefix="miscls")
        performance = _eval_amb_detection(miscls_results, miscls_targets, performance, prefix="miscls")

    with open(result_dir / "amb_task_performance.json", "w") as f:
        json.dump(performance, f, indent=4)
    print(f"Saved amb_task results to {result_dir / 'amb_task_performance.json'}")

    val_ds.negative_label = False
    test_ds.negative_label = False
    return performance


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_methods_to_run(cfg: DictConfig) -> List[str]:
    methods_cfg = cfg.get("methods_to_run", [])
    available_methods = MethodFactory.get_available_methods()
    all_methods_tokens = {"all_methods", "all"}

    if isinstance(methods_cfg, str):
        raw = methods_cfg.strip()
        if raw.lower() in all_methods_tokens:
            methods = list(available_methods)
        else:
            methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    else:
        methods = [str(m).strip().lower() for m in methods_cfg]

    if any(m in all_methods_tokens for m in methods):
        methods = list(available_methods)

    if not methods:
        default_method = str(cfg.method.name).strip().lower()
        if default_method in all_methods_tokens:
            methods = list(available_methods)
        else:
            methods = [default_method]

    unique_methods: List[str] = []
    for method_name in methods:
        if method_name not in unique_methods:
            unique_methods.append(method_name)

    available_methods_set = set(available_methods)
    unknown_methods = [m for m in unique_methods if m not in available_methods_set]
    if unknown_methods:
        raise ValueError(
            f"Unknown method(s): {unknown_methods}. Available methods: {sorted(available_methods_set)}"
        )

    return unique_methods


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))

    set_random_seed(int(cfg.seed))

    os.makedirs(cfg.data.root, exist_ok=True)

    if "distortion_pattern" not in cfg.experiment:
        raise ValueError("cfg.experiment.distortion_pattern must be set explicitly.")
    distortion_pattern = str(cfg.experiment.distortion_pattern)

    # The ambiguity task uses its own loader logic; for other patterns build normally.
    is_amb = (distortion_pattern == "amb")

    adapter = get_dataset_adapter(cfg)
    base_train_loader, base_val_loader, eval_loaders, level_names = adapter.build_loaders(
        cfg,
        distortion_pattern=distortion_pattern if not is_amb else "plain",
    )

    dataset_name = str(cfg.dataset.name)
    experiment_name = str(cfg.experiment.name)
    run_id = str(cfg.output.get("run_id", "run"))
    project_root = Path(HydraConfig.get().runtime.cwd)

    methods_to_run = _resolve_methods_to_run(cfg)
    comparison_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    auroc_summary: Dict[str, Dict[str, float]] = {}

    for method_name in methods_to_run:
        print({"method": method_name, "status": "start"})
        set_random_seed(int(cfg.seed))

        method_cfg = MethodFactory.load_method_config(cfg, method_name)
        method = MethodFactory.create(method_cfg)

        model_dir = project_root / "models" / dataset_name / method_name
        hydra_out = Path(HydraConfig.get().runtime.output_dir)
        result_dir = project_root / "results" / experiment_name / method_name

        # ── Skip if this method's results already exist ────────────────────
        _done_marker = result_dir / ("amb_task_performance.json" if is_amb else "uncertainties_summary.json")
        if _done_marker.exists():
            print({"method": method_name, "status": "skipped (results exist)", "path": str(result_dir)})
            if not is_amb:
                with open(_done_marker) as _f:
                    comparison_summary[method_name] = json.load(_f)
            continue
        # Mirror results path under plots/: plots/YYYY-MM-DD/experiment_name/HH-MM-SS/method_name
        plot_dir = Path("plots") / Path(*hydra_out.parts[-3:]) / method_name
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_uncertainties: Dict[str, Dict[str, object]] = {}
        model_path = model_dir / f"base_model_{method_cfg.model.name}.pt"

        if model_path.exists():
            method.load_model(str(model_path), train_loader=base_train_loader, val_loader=base_val_loader)
            print({"method": method_name, "model": "loaded", "path": str(model_path)})
        else:
            method.train_model(
                base_train_loader,
                base_val_loader,
                epochs=int(method_cfg.experiment.epochs),
                lr=float(method_cfg.experiment.lr),
            )
            method.save_model(str(model_path))
            print({"method": method_name, "model": "trained", "path": str(model_path)})

        # ── Ambiguity task ────────────────────────────────────────────────
        if is_amb:
            amb_plot_dir = plot_dir / "amb"
            run_ambiguous_uncertainty_task(
                cfg=cfg,
                method=method,
                val_loader=base_val_loader,
                test_loader=list(eval_loaders.values())[0],
                result_dir=result_dir,
                plot_dir=amb_plot_dir,
                num_samples=int(cfg.get("num_samples", -1)),
            )
            comparison_summary[method_name] = {}
            continue

        # Measure uncertainty on clean validation set (ID baseline)
        valid_uncertainty_path = result_dir / "valid_uncertainties.pkl"
        if valid_uncertainty_path.exists():
            with open(valid_uncertainty_path, "rb") as f:
                valid_uncertainties = pickle.load(f)
            print({"method": method_name, "split": "val", "uncertainty": "loaded"})
        else:
            valid_uncertainties = method.measure_uncertainty(base_val_loader)
            with open(valid_uncertainty_path, "wb") as f:
                pickle.dump(valid_uncertainties, f)
            print({"method": method_name, "split": "val", "uncertainty": "computed"})

        # Measure uncertainty on each distortion level (OOD)
        ood_uncertainties: Dict[str, dict] = {}
        for level in level_names:
            uncertainty_path = result_dir / f"ood_uncertainty_{level}.pkl"

            if uncertainty_path.exists():
                with open(uncertainty_path, "rb") as f:
                    uncertainty = pickle.load(f)
                print({"method": method_name, "level": level, "uncertainty": "loaded"})
            else:
                uncertainty = method.measure_uncertainty(eval_loaders[level])
                with open(uncertainty_path, "wb") as f:
                    pickle.dump(uncertainty, f)
                print({"method": method_name, "level": level, "uncertainty": "computed"})

            ood_uncertainties[level] = uncertainty
            plot_uncertainties[level] = uncertainty

        uncertainty_keys = ("total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty")
        method_summary = {
            level: {
                k: float(torch.mean(plot_uncertainties[level][k]).item())
                if isinstance(plot_uncertainties[level][k], torch.Tensor)
                else float(np.mean(plot_uncertainties[level][k]))
                for k in uncertainty_keys
            }
            for level in level_names
        }

        with open(result_dir / "uncertainties_summary.json", "w", encoding="utf-8") as f:
            json.dump(method_summary, f, indent=2)

        trend_plt = trend(plot_uncertainties)
        trend_plt.tight_layout()
        trend_plt.savefig(plot_dir / "trend.png")
        trend_plt.clf()

        # ── OOD subgroup evaluation ───────────────────────────────────
        _subgroup_patterns = {"by_age", "by_disease_count", "by_gender", 
                              "age", "ink", "drop", "hair", "skin_tone"}
        if distortion_pattern in _subgroup_patterns:
            run_ood_subgroup_task(
                cfg=cfg,
                method=method,
                eval_loaders=eval_loaders,
                level_names=level_names,
                result_dir=result_dir / "ood_subgroup",
                plot_dir=plot_dir / "ood_subgroup",
            )            
            
        _decomp_uq = None
        if distortion_pattern == "mnist_uncertainty_decomp_blur":
            _decomp_uq = "aleatoric_uncertainty"
        elif distortion_pattern == "mnist_uncertainty_decomp_fracture":
            _decomp_uq = "epistemic_uncertainty"
        if _decomp_uq is not None:
            decomp_perf = run_uncertainty_decomposition(
                cfg=cfg,
                method=method,
                eval_loaders=eval_loaders,
                level_names=level_names,
                expected_uq_type=_decomp_uq,
                result_dir=result_dir / "sensitivity",
                plot_dir=plot_dir / "sensitivity",
            )
            auroc_summary[method_name] = {
                k: v for k, v in decomp_perf.items() if k.startswith("auroc_")
            }


        comparison_summary[method_name] = method_summary
        print({"method": method_name, "status": "done"})

    if len(methods_to_run) > 1:
        comparison_path = project_root / "results" / experiment_name / "method_comparison_summary.json"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_summary, f, indent=2)
        print({"comparison_summary": str(comparison_path)})

    # ── Cross-method AUROC comparison ─────────────────────────────────────
    if auroc_summary:
        cmp_plot_dir = Path("plots") / Path(*Path(HydraConfig.get().runtime.output_dir).parts[-3:]) / "comparison"
        cmp_plot_dir.mkdir(parents=True, exist_ok=True)
        cmp_result_dir = project_root / "results" / experiment_name
        cmp_result_dir.mkdir(parents=True, exist_ok=True)

        with open(cmp_result_dir / "auroc_summary.json", "w", encoding="utf-8") as f:
            json.dump(auroc_summary, f, indent=2)

        method_names_with_auroc = list(auroc_summary.keys())
        colors = plt.cm.tab10.colors

        # Detect which uq_key was used (same for all methods in a run)
        _sample_keys = list(next(iter(auroc_summary.values())).keys())
        if any("aleatoric_uncertainty" in k for k in _sample_keys):
            _cmp_uq_key = "aleatoric_uncertainty"
        else:
            _cmp_uq_key = "epistemic_uncertainty"

        # Per-level AUROC line plot (one line per method)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(level_names))
        for i, mname in enumerate(method_names_with_auroc):
            per_level = [auroc_summary[mname].get(f"auroc_{_cmp_uq_key}_{lvl}", float("nan")) for lvl in level_names]
            ax.plot(x, per_level, marker="o", label=mname, color=colors[i % len(colors)])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
        ax.set_xticks(x)
        ax.set_xticklabels(level_names, rotation=30, ha="right")
        ax.set_ylabel("AUROC  (dominant uncertainty vs not)")
        ax.set_title(f"Per-level AUROC — all methods  [{distortion_pattern}]")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(cmp_plot_dir / "auroc_per_level_comparison.png", bbox_inches="tight")
        plt.close(fig)

        # Mean AUROC bar chart
        mean_aurocs = [auroc_summary[m].get("auroc_mean", float("nan")) for m in method_names_with_auroc]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(np.arange(len(method_names_with_auroc)), mean_aurocs,
               color=[colors[i % len(colors)] for i in range(len(method_names_with_auroc))])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
        ax.set_xticks(np.arange(len(method_names_with_auroc)))
        ax.set_xticklabels(method_names_with_auroc, rotation=30, ha="right")
        ax.set_ylabel("Mean AUROC")
        ax.set_title(f"Mean AUROC across levels — all methods  [{distortion_pattern}]")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(cmp_plot_dir / "auroc_mean_comparison.png", bbox_inches="tight")
        plt.close(fig)
        print({"auroc_comparison": str(cmp_plot_dir)})

    print({"status": "done", "methods": methods_to_run, "levels": level_names, "output_dir": os.getcwd() + "/results"})


if __name__ == "__main__":
    main()

    