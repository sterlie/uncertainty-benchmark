from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader


LoaderBundle = Tuple[DataLoader, DataLoader, Dict[str, DataLoader], list[str]]


class DatasetExperimentAdapter(ABC):
    """Dataset adapter used by experiment runners to build train/eval loaders."""

    @abstractmethod
    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        """Return (train_loader, val_loader, eval_loaders, level_names)."""

    def supports_cross_validation(self) -> bool:
        """Whether this dataset adapter can emit CV folds in future extensions."""
        return False
