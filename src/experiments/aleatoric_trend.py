import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.methods.method_factory import MethodFactory
from src.datasets.mnist import build_mnist_loaders


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_trend(uncertainties: Dict[str, Dict[str, np.ndarray]], ordered_levels: List[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    keys = ["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"]

    for axis, key in zip(axes, keys):
        means = []
        for level in ordered_levels:
            means.append(float(np.mean(uncertainties[level][key])))
        axis.plot(ordered_levels, means, marker="o")
        axis.set_title(key)
        axis.set_xlabel("severity")
        axis.set_ylabel("mean uncertainty")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plot_uncertainty_view(uncertainty_dict: Dict[str, object]) -> Dict[str, np.ndarray]:
    return {
        "total_uncertainty": _to_numpy(uncertainty_dict["total_uncertainty"]),
        "aleatoric_uncertainty": _to_numpy(uncertainty_dict["aleatoric_uncertainty"]),
        "epistemic_uncertainty": _to_numpy(uncertainty_dict["epistemic_uncertainty"]),
    }


def _resolve_methods_to_run(cfg: DictConfig) -> List[str]:
    methods_cfg = cfg.get("methods_to_run", [])

    if isinstance(methods_cfg, str):
        methods = [m.strip() for m in methods_cfg.split(",") if m.strip()]
    else:
        methods = [str(m) for m in methods_cfg]

    if not methods:
        methods = [str(cfg.method.name)]

    unique_methods: List[str] = []
    for method_name in methods:
        if method_name not in unique_methods:
            unique_methods.append(method_name)

    available_methods = set(MethodFactory.get_available_methods())
    unknown_methods = [m for m in unique_methods if m not in available_methods]
    if unknown_methods:
        raise ValueError(
            f"Unknown method(s): {unknown_methods}. Available methods: {sorted(available_methods)}"
        )

    return unique_methods


def _method_cfg_copy(cfg: DictConfig, method_name: str) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_copy.method.name = method_name
    return cfg_copy


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))

    set_random_seed(int(cfg.seed))

    os.makedirs(cfg.data.root, exist_ok=True)

    # Build data loaders with blur distortions
    base_train_loader, base_val_loader, eval_loaders, level_names = build_mnist_loaders(cfg, distortion_pattern="blur")

    methods_to_run = _resolve_methods_to_run(cfg)
    comparison_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method_name in methods_to_run:
        print({"method": method_name, "status": "start"})
        set_random_seed(int(cfg.seed))

        method_cfg = _method_cfg_copy(cfg, method_name)
        method = MethodFactory.create(method_cfg)

        model_dir = Path("models") / method_name
        result_dir = Path("results") / method_name
        plot_dir = Path("plots") / method_name
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_uncertainties: Dict[str, Dict[str, np.ndarray]] = {}
        model_path = model_dir / f"base_model_{method_cfg.model.name}.pt"

        if model_path.exists():
            method.load_model(str(model_path))
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

        for level in level_names:
            uncertainty_path = result_dir / f"validset_uncertainty_{level}.pkl"

            if uncertainty_path.exists():
                with open(uncertainty_path, "rb") as f:
                    uncertainty = pickle.load(f)
                print({"method": method_name, "level": level, "uncertainty": "loaded"})
            else:
                uncertainty = method.measure_uncertainty(eval_loaders[level])
                with open(uncertainty_path, "wb") as f:
                    pickle.dump(uncertainty, f)
                print({"method": method_name, "level": level, "uncertainty": "computed"})

            plot_uncertainties[level] = _plot_uncertainty_view(uncertainty)

        method_summary = {
            level: {
                k: float(np.mean(v))
                for k, v in plot_uncertainties[level].items()
            }
            for level in level_names
        }

        with open(result_dir / "uncertainties_summary.json", "w", encoding="utf-8") as f:
            json.dump(method_summary, f, indent=2)

        plot_trend(plot_uncertainties, level_names, plot_dir / "trend.png")
        comparison_summary[method_name] = method_summary
        print({"method": method_name, "status": "done"})

    if len(methods_to_run) > 1:
        comparison_path = Path("results") / "method_comparison_summary.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_summary, f, indent=2)
        print({"comparison_summary": str(comparison_path)})

    print({"status": "done", "methods": methods_to_run, "levels": level_names, "output_dir": os.getcwd() + "/results"})


if __name__ == "__main__":
    main()