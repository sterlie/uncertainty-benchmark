from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader


LoaderBundle = Tuple[DataLoader, DataLoader, Dict[str, DataLoader], list[str]]


def subset_df(df, subset_size):
    """Return the first *subset_size* rows of *df*, or all rows when *subset_size* is None."""
    if subset_size is None:
        return df
    n = min(int(subset_size), len(df))
    return df.iloc[:n].reset_index(drop=True)


class DatasetExperimentAdapter(ABC):
    """Dataset adapter used by experiment runners to build train/eval loaders."""

    @abstractmethod
    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        """Return (train_loader, val_loader, eval_loaders, level_names)."""

    def supports_cross_validation(self) -> bool:
        """Whether this dataset adapter can emit CV folds in future extensions."""
        return False
