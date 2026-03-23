from abc import ABC, abstractmethod
from typing import Dict, Type, Any

import numpy as np
import torch
from torch.utils.data import Dataset #as TorchDataset
#from morphomnist import morpho, perturb

class DatasetFactory:
    """Factory for creating dataset instances based on config."""

    _registry: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str, dataset_class: Type[Dataset]):
        """Register a dataset class with a name."""
        cls._registry[name] = dataset_class

    @classmethod
    def create(cls, config, is_training=True, perturbation=None) -> Dataset:
        """Create a dataset instance based on config."""
        dataset_name = config.dataset.name.lower()

        if dataset_name not in cls._registry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {list(cls._registry.keys())}")

        return cls._registry[dataset_name](config, is_training, perturbation)

    @classmethod
    def get_available_datasets(cls) -> list:
        """Get list of available dataset names."""
        return list(cls._registry.keys())


def register_dataset(name: str):
    """Decorator to register a dataset class."""
    def decorator(dataset_class: Type[Dataset]):
        DatasetFactory.register(name, dataset_class)
        return dataset_class
    return decorator