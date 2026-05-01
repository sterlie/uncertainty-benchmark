from __future__ import annotations

from omegaconf import DictConfig

from src.experiments.datasets.base import DatasetExperimentAdapter
from src.experiments.datasets.isic_adapter import ISICExperimentAdapter
from src.experiments.datasets.mnist_adapter import MNISTExperimentAdapter
from src.experiments.datasets.chest_adapter import ChestExperimentAdapter

_ADAPTERS = {
    "mnist": MNISTExperimentAdapter,
    "isic": ISICExperimentAdapter,
    "chexpert": ChestExperimentAdapter,
    "nih": ChestExperimentAdapter,
    "vin": ChestExperimentAdapter,
}


def get_dataset_adapter(cfg: DictConfig) -> DatasetExperimentAdapter:
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(
            f"No experiment adapter registered for dataset '{dataset_name}'. "
            f"Available adapters: {available}"
        )
    return _ADAPTERS[dataset_name]()
