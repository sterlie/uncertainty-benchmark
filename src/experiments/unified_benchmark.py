import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from src.experiments.datasets import get_dataset_adapter
from src.utils.visualization import moving_out_of_distribution_compare


UNCERTAINTY_TYPES = (
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "total_uncertainty",
)


def _resolve_distortion_pattern(cfg: DictConfig) -> str:
    """Resolve distortion pattern from experiment config."""
    explicit = cfg.experiment.get("distortion_pattern", None)
    return str(explicit) if explicit else "blur"


def _build_palette(display_names: List[str]) -> Dict[str, tuple]:
    colors = plt.get_cmap("tab10").colors
    return {name: colors[i % len(colors)] for i, name in enumerate(display_names)}


def _build_subexperiment_cfg(
    base_cfg: DictConfig,
    template_name: str,
    fixed_run_id: str,
) -> DictConfig:
    """
    Build a config for a sub-experiment template while keeping the top-level
    experiment name unchanged (so all artifacts remain under unified_benchmark).
    """
    cfg_copy = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "config" / "experiment" / f"{template_name}.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Sub-experiment config not found: {template_path}")

    template_cfg = OmegaConf.load(template_path)
    unified_name = str(base_cfg.experiment.name)
    template_cfg.name = unified_name
    cfg_copy.experiment = template_cfg
    cfg_copy.output.run_id = fixed_run_id
    return cfg_copy


def _load_ood_uncertainties_from_results(
    dataset_name: str,
    experiment_name: str,
    distortion_pattern: str,
    run_id: str,
    methods: List[str],
    levels: List[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    """
    Load cached OOD uncertainties from results directory.
    Assumes structure:
    results/{dataset_name}/{experiment_name}/{distortion_pattern}/{method}/{run_id}/validset_uncertainty_{level}.pkl
    """
    distortion_family = f"{dataset_name}_{distortion_pattern}"
    ood_uncertainties: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {
        distortion_family: {}
    }

    for method in methods:
        result_dir = Path("results") / dataset_name / experiment_name / distortion_pattern / method / run_id
        if not result_dir.exists():
            print(f"Warning: result dir not found for {method}: {result_dir}")
            continue

        method_uncertainties: Dict[str, Dict[str, object]] = {}
        for level in levels:
            cache_file = result_dir / f"validset_uncertainty_{level}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    method_uncertainties[level] = pickle.load(f)
            else:
                print(f"Warning: cache file not found: {cache_file}")

        ood_uncertainties[distortion_family][method] = method_uncertainties

    return ood_uncertainties


def _split_id_ood_uncertainties(
    uncertainty_per_distortion: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, Dict[str, object]]], Dict[str, str]]:
    """
    Split cached uncertainties into:
    - id_uncertainty: one entry per method (prefers 'plain' level)
    - ood_uncertainty: all non-ID levels per method
    - id_level_by_method: selected ID level label for traceability
    """
    id_uncertainty: Dict[str, Dict[str, object]] = {}
    ood_uncertainty: Dict[str, Dict[str, Dict[str, object]]] = {}
    id_level_by_method: Dict[str, str] = {}

    if not uncertainty_per_distortion:
        return id_uncertainty, ood_uncertainty, id_level_by_method

    distortion_key = next(iter(uncertainty_per_distortion.keys()))
    method_payload = uncertainty_per_distortion[distortion_key]

    for method_name, level_map in method_payload.items():
        if not level_map:
            continue

        id_level = "plain" if "plain" in level_map else next(iter(level_map.keys()))
        id_level_by_method[method_name] = id_level
        id_uncertainty[method_name] = level_map[id_level]

        ood_levels = {lvl: payload for lvl, payload in level_map.items() if lvl != id_level}
        ood_uncertainty[method_name] = ood_levels

    return id_uncertainty, ood_uncertainty, id_level_by_method


def _save_id_ood_artifacts(
    dataset_name: str,
    experiment_name: str,
    distortion_pattern: str,
    run_id: str,
    id_uncertainty: Dict[str, Dict[str, object]],
    ood_uncertainty: Dict[str, Dict[str, Dict[str, object]]],
    id_level_by_method: Dict[str, str],
) -> None:
    artifact_dir = (
        Path("results")
        / dataset_name
        / experiment_name
        / distortion_pattern
        / "comparison"
        / run_id
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "id_uncertainty.pkl", "wb") as f:
        pickle.dump(id_uncertainty, f)

    with open(artifact_dir / "ood_uncertainty.pkl", "wb") as f:
        pickle.dump(ood_uncertainty, f)

    with open(artifact_dir / "id_level_by_method.pkl", "wb") as f:
        pickle.dump(id_level_by_method, f)


def _postprocess_pattern(
    cfg: DictConfig,
    methods: List[str],
) -> None:
    dataset_name = str(cfg.dataset.name)
    experiment_name = str(cfg.experiment.name)
    distortion_pattern = _resolve_distortion_pattern(cfg)
    run_id = str(cfg.output.get("run_id", "run"))

    adapter = get_dataset_adapter(cfg)
    _, _, _, level_names = adapter.build_loaders(cfg, distortion_pattern=distortion_pattern)
    legend_map = {method: method for method in methods}
    palette = _build_palette(list(legend_map.values()))

    ood_uncertainties = _load_ood_uncertainties_from_results(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        distortion_pattern=distortion_pattern,
        run_id=run_id,
        methods=methods,
        levels=level_names,
    )

    experiment_key = list(ood_uncertainties.keys())[0] if ood_uncertainties else None
    if not experiment_key or not ood_uncertainties[experiment_key]:
        print(
            "ERROR: No result data found for any method. Results directory should be: "
            f"results/{dataset_name}/{experiment_name}/{distortion_pattern}/{{method}}/{run_id}/"
        )
        print("Ensure the experiment ran successfully and generated validset_uncertainty_*.pkl files.")
        return

    id_uncertainty, ood_only_uncertainty, id_level_by_method = _split_id_ood_uncertainties(
        ood_uncertainties
    )
    _save_id_ood_artifacts(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        distortion_pattern=distortion_pattern,
        run_id=run_id,
        id_uncertainty=id_uncertainty,
        ood_uncertainty=ood_only_uncertainty,
        id_level_by_method=id_level_by_method,
    )

    print(f"Generating comparison plots for OOD pattern: {distortion_pattern}")
    for uncertainty_type in UNCERTAINTY_TYPES:
        try:
            fig = moving_out_of_distribution_compare(
                ood_uncertainties,
                legend_map=legend_map,
                palette=palette,
                uncertainty_type=uncertainty_type,
            )
            plot_dir = (
                Path("plots")
                / dataset_name
                / experiment_name
                / distortion_pattern
                / "comparison"
                / run_id
            )
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"moving_out_of_dist_{uncertainty_type}.png")
            plt.close()
            print(f"Saved: {plot_dir / f'moving_out_of_dist_{uncertainty_type}.png'}")
        except Exception as e:
            print(f"Warning: Failed to generate {uncertainty_type} plot for {distortion_pattern}: {e}")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Unified benchmark orchestrator.
    
    Runs the appropriate experiment (aleatoric_trend or epistemic_trend) based on config,
    then adds moving_out_of_distribution_compare plots.
    """
    print("Unified Benchmark Orchestrator")
    print(f"Experiment: {cfg.experiment.name}")

    methods_cfg = cfg.get("methods_to_run", [])
    if isinstance(methods_cfg, str):
        methods = [m.strip() for m in methods_cfg.split(",") if m.strip()]
    elif isinstance(methods_cfg, list):
        methods = [str(m) for m in methods_cfg]
    else:
        methods = [str(cfg.method.name)]

    if not methods:
        methods = [str(cfg.method.name)]

    exp_name = str(cfg.experiment.get("name", "")).lower()
    unified_run_id = str(cfg.output.get("run_id", "run"))

    from src.experiments.aleatoric_trend import main as aleatoric_main
    from src.experiments.epistemic_trend import main as epistemic_main

    if "unified" in exp_name:
        sub_runs = [
            ("aleatoric_trend", aleatoric_main),
            ("epistemic_trend", epistemic_main),
        ]

        for template_name, runner in sub_runs:
            print(f"Running sub-experiment: {template_name}")
            sub_cfg = _build_subexperiment_cfg(cfg, template_name, unified_run_id)
            runner(sub_cfg)
            _postprocess_pattern(sub_cfg, methods)
    elif "aleatoric" in exp_name:
        print("Running aleatoric_trend experiment...")
        aleatoric_main(cfg)
        _postprocess_pattern(cfg, methods)
    elif "epistemic" in exp_name:
        print("Running epistemic_trend experiment...")
        epistemic_main(cfg)
        _postprocess_pattern(cfg, methods)
    else:
        print(f"Unknown experiment type: {exp_name}")
        return

    print(
        {
            "status": "completed",
            "experiment": cfg.experiment.name,
            "methods": methods,
            "extra_plots": "moving_out_of_distribution_compare per OOD pattern",
        }
    )


if __name__ == "__main__":
    main()
