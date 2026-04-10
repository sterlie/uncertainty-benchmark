from __future__ import annotations

from omegaconf import DictConfig

from src.datasets.mnist import build_mnist_loaders
from src.experiments.datasets.base import DatasetExperimentAdapter, LoaderBundle


class MNISTExperimentAdapter(DatasetExperimentAdapter):
    """Adapter for MNIST-family runs used by current aleatoric/epistemic experiments."""

    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        return build_mnist_loaders(cfg, distortion_pattern=distortion_pattern)
