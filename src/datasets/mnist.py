"""Generalized MNIST dataset loaders supporting multiple distortion patterns."""
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms


def _use_subset(dataset, subset_size):
    """Optionally limit dataset to a subset size."""
    if subset_size is None:
        return dataset
    subset_size = min(int(subset_size), len(dataset))
    return Subset(dataset, list(range(subset_size)))


def _mnist_transform(normalize: bool, blur_kernel: int | None = None, blur_sigma: float | None = None):
    """Build MNIST transform pipeline with optional blur."""
    tfms = [transforms.ToTensor()]
    if blur_kernel is not None and blur_sigma is not None:
        tfms.append(transforms.GaussianBlur(kernel_size=int(blur_kernel), sigma=float(blur_sigma)))
    if normalize:
        tfms.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(tfms)


def _build_plain_datasets(
    root: str,
    normalize: bool,
    train_subset_size,
    test_subset_size,
):
    train_plain = datasets.MNIST(
        root=root, train=True, download=True, transform=_mnist_transform(normalize)
    )
    val_plain = datasets.MNIST(
        root=root, train=False, download=True, transform=_mnist_transform(normalize)
    )

    train_plain = _use_subset(train_plain, train_subset_size)
    val_plain = _use_subset(val_plain, test_subset_size)
    return train_plain, val_plain


def _build_blur_loaders(
    root: str,
    batch_size: int,
    normalize: bool,
    severity_levels,
    train_subset_size,
    test_subset_size,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader], List[str]]:
    """Build clean training loaders and severity-specific blurred evaluation loaders."""
    train_plain, val_plain = _build_plain_datasets(
        root=root,
        normalize=normalize,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
    )

    clean_train_loader = DataLoader(
        train_plain,
        batch_size=batch_size,
        shuffle=True,
    )
    clean_val_loader = DataLoader(
        val_plain,
        batch_size=batch_size,
        shuffle=False,
    )

    eval_loaders: Dict[str, DataLoader] = {}
    level_names: List[str] = []

    # For each severity level, evaluate on clean + blurred validation data.
    for level in severity_levels:
        level_name = str(level.name)
        level_names.append(level_name)

        blur_kernel = int(level.kernel)
        blur_sigma = float(level.sigma)

        train_blur = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=_mnist_transform(normalize, blur_kernel=blur_kernel, blur_sigma=blur_sigma),
        )
        val_blur = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=_mnist_transform(normalize, blur_kernel=blur_kernel, blur_sigma=blur_sigma),
        )

        val_blur = _use_subset(val_blur, test_subset_size)

        val_mix = ConcatDataset([val_plain, val_blur])

        eval_loaders[level_name] = DataLoader(
            val_mix,
            batch_size=batch_size,
            shuffle=False,
        )

    return clean_train_loader, clean_val_loader, eval_loaders, level_names


def build_mnist_loaders(
    cfg: DictConfig, distortion_pattern: str = "blur"
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader], List[str]]:
    """
    Generic MNIST loader builder supporting multiple distortion patterns.

    Args:
        cfg: Hydra config with dataset, experiment, and data settings
        distortion_pattern: Type of distortion ("blur" or "none")

    Returns:
        Tuple of (train_loaders, val_loaders, level_names)
    """
    root = cfg.data.root
    batch_size = int(cfg.dataset.batch_size)
    normalize = bool(cfg.experiment.get("normalize", True))
    train_subset = cfg.dataset.get("train_subset", None)
    test_subset = cfg.dataset.get("test_subset", None)

    if distortion_pattern == "blur":
        severity_levels = cfg.experiment.severity_levels
        return _build_blur_loaders(
            root=root,
            batch_size=batch_size,
            normalize=normalize,
            severity_levels=severity_levels,
            train_subset_size=train_subset,
            test_subset_size=test_subset,
        )
    else:
        raise ValueError(f"Unknown distortion pattern: {distortion_pattern}")
