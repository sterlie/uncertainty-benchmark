import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


from src.experiments.datasets import get_dataset_adapter
from src.methods.method_factory import MethodFactory
from src.utils.visualization import trend



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
		if raw in all_methods_tokens:
			methods = list(available_methods)
		else:
			methods = [m.strip() for m in raw.split(",") if m.strip()]
	else:
		methods = [str(m) for m in methods_cfg]

	if any(m in all_methods_tokens for m in methods):
		methods = list(available_methods)

	if not methods:
		default_method = str(cfg.method.name)
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
	adapter = get_dataset_adapter(cfg)
	base_train_loader, base_val_loader, eval_loaders, level_names = adapter.build_loaders(
		cfg,
		distortion_pattern=distortion_pattern,
	)

	dataset_name = str(cfg.dataset.name)
	experiment_name = str(cfg.experiment.name)
	run_id = str(cfg.output.get("run_id", "run"))

	methods_to_run = _resolve_methods_to_run(cfg)
	comparison_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

	for method_name in methods_to_run:
		print({"method": method_name, "status": "start"})
		set_random_seed(int(cfg.seed))

		method_cfg = MethodFactory.load_method_config(cfg, method_name)
		method = MethodFactory.create(method_cfg)

		model_dir = Path("models") / dataset_name / method_name
		result_dir = Path(HydraConfig.get().runtime.output_dir)
		plot_dir = Path("plots") / dataset_name / experiment_name / distortion_pattern / method_name / run_id
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

		for level in level_names:
			uncertainty_path = result_dir / f"validset_uncertainty_level_{level}.pkl"

			if uncertainty_path.exists():
				with open(uncertainty_path, "rb") as f:
					uncertainty = pickle.load(f)
				print({"method": method_name, "level": level, "uncertainty": "loaded"})
			else:
				uncertainty = method.measure_uncertainty(eval_loaders[level])
				with open(uncertainty_path, "wb") as f:
					pickle.dump(uncertainty, f)
				print({"method": method_name, "level": level, "uncertainty": "computed"})

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
		trend_plt.close()
		comparison_summary[method_name] = method_summary
		print({"method": method_name, "status": "done"})

	if len(methods_to_run) > 1:
		comparison_path = Path("results") / dataset_name / experiment_name / distortion_pattern / run_id / "method_comparison_summary.json"
		comparison_path.parent.mkdir(parents=True, exist_ok=True)
		with open(comparison_path, "w", encoding="utf-8") as f:
			json.dump(comparison_summary, f, indent=2)
		print({"comparison_summary": str(comparison_path)})

	print({"status": "done", "methods": methods_to_run, "fracture levels": level_names, "output_dir": os.getcwd() + "/results"})


if __name__ == "__main__":
	main()
