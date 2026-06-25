"""Generalized MNIST dataset loaders supporting multiple distortion patterns."""
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms
from src.datasets import morpho_mnist as morpho_mnist_module

def _use_subset(dataset, subset_size):
    """Optionally limit dataset to a subset size."""
    if subset_size is None:
        return dataset
    subset_size = min(int(subset_size), len(dataset))
    return Subset(dataset, list(range(subset_size)))


def _mnist_transform(normalize: bool, blur_kernel: int | None = None, blur_sigma: float | None = None, blur_prob: float | None = None):
    """Build MNIST transform pipeline with optional blur."""
    tfms = [transforms.ToTensor()]
    if blur_kernel is not None and blur_sigma is not None:
        blur = transforms.GaussianBlur(kernel_size=int(blur_kernel), sigma=float(blur_sigma))
        if blur_prob is not None:
            tfms.append(transforms.RandomApply([blur], p=float(blur_prob)))
        else:
            tfms.append(blur)
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
    _, val_plain = _build_plain_datasets(
        root=root,
        normalize=normalize,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
    )
    train_blurred = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=_mnist_transform(normalize, blur_kernel=5, blur_sigma=1, blur_prob=0.2),
    )
    train_blurred = _use_subset(train_blurred, train_subset_size)

    clean_train_loader = DataLoader(
        train_blurred,
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

        val_blur = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=_mnist_transform(normalize, blur_kernel=blur_kernel, blur_sigma=blur_sigma),
        )

        val_blur = _use_subset(val_blur, test_subset_size)

        eval_loaders[level_name] = DataLoader(
            val_blur,
            batch_size=batch_size,
            shuffle=False,
        )

    return clean_train_loader, clean_val_loader, eval_loaders, level_names


def _build_fracture_loaders(
    root: str,
    batch_size: int,
    normalize: bool,
    severity_levels,
    train_subset_size,
    test_subset_size,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader], List[str]]:
    """Build clean MNIST train/val loaders and fracture-distorted eval loaders."""

    MorphoMNISTDataset = morpho_mnist_module.MorphoMNISTDataset
    perturb = morpho_mnist_module.perturb

    _, val_plain = _build_plain_datasets(
        root=root,
        normalize=normalize,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
    )

    raw_train = datasets.MNIST(root=root, train=True, download=True)
    train_images = raw_train.data.numpy()
    train_labels = raw_train.targets.numpy()
    if train_subset_size is not None:
        train_size = min(int(train_subset_size), len(train_images))
        train_images = train_images[:train_size]
        train_labels = train_labels[:train_size]
    train_dataset = MorphoMNISTDataset(
        train_images,
        train_labels,
        perturbation=perturb.Identity(),
        transform=_mnist_transform(normalize, blur_kernel=5, blur_sigma=1, blur_prob=0.2),
    )

    clean_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    clean_val_loader = DataLoader(
        val_plain,
        batch_size=batch_size,
        shuffle=False,
    )

    raw_val = datasets.MNIST(root=root, train=False, download=True)
    val_images = raw_val.data.numpy()
    val_labels = raw_val.targets.numpy()
    if test_subset_size is not None:
        subset_size = min(int(test_subset_size), len(val_images))
        val_images = val_images[:subset_size]
        val_labels = val_labels[:subset_size]

    eval_loaders: Dict[str, DataLoader] = {}
    level_names: List[str] = []
    base_transform = _mnist_transform(normalize)

    for level in range(1, int(severity_levels) + 1):
        if isinstance(level, (int, float)):
            fracture_count = int(level)
            level_name = str(fracture_count)
        else:
            fracture_count = int(level.get("fractures", 0))
            level_name = str(level.get("name", fracture_count))
        level_names.append(level_name)

        if fracture_count <= 0:
            eval_dataset = MorphoMNISTDataset(
                val_images,
                val_labels,
                perturbation=None,
                transform=base_transform,
            )
        else:
            eval_dataset = MorphoMNISTDataset(
                val_images,
                val_labels,
                perturbation=perturb.Fracture(num_frac=fracture_count),
                transform=base_transform,
            )

        eval_loaders[level_name] = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    return clean_train_loader, clean_val_loader, eval_loaders, level_names


def _build_thinning_loaders(
    root: str,
    batch_size: int,
    normalize: bool,
    severity_levels,
    train_subset_size,
    test_subset_size,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader], List[str]]:
    """Build clean MNIST train/val loaders and thinning-distorted eval loaders."""

    MorphoMNISTDataset = morpho_mnist_module.MorphoMNISTDataset
    perturb = morpho_mnist_module.perturb

    _, val_plain = _build_plain_datasets(
        root=root,
        normalize=normalize,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
    )
    train_blurred = datasets.MNIST(
        root=root, train=True, download=True,
        transform=_mnist_transform(normalize, blur_kernel=5, blur_sigma=1, blur_prob=0.2),
    )
    train_blurred = _use_subset(train_blurred, train_subset_size)

    clean_train_loader = DataLoader(
        train_blurred,
        batch_size=batch_size,
        shuffle=True)
    clean_val_loader = DataLoader(
        val_plain, 
        batch_size=batch_size, 
        shuffle=False)

    raw_val = datasets.MNIST(root=root, train=False, download=True)
    val_images = raw_val.data.numpy()
    val_labels = raw_val.targets.numpy()
    if test_subset_size is not None:
        subset_size = min(int(test_subset_size), len(val_images))
        val_images = val_images[:subset_size]
        val_labels = val_labels[:subset_size]

    eval_loaders: Dict[str, DataLoader] = {}
    level_names: List[str] = []
    base_transform = _mnist_transform(normalize)

    if isinstance(severity_levels, int):
        raise ValueError('For thinning, specify distortion levels as a list, e.g. [0.1, 0.3, 0.5, 0.7, 0.9]')
    amounts = [float(a) for a in severity_levels]

    for amount in amounts:
        level_name = str(amount)
        level_names.append(level_name)

        eval_dataset = MorphoMNISTDataset(
            val_images,
            val_labels,
            perturbation=perturb.Thinning(amount=amount),
            transform=base_transform,
        )

        eval_loaders[level_name] = DataLoader(
            eval_dataset,
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
        distortion_pattern: Type of distortion ("blur" or "fracture")

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
    if distortion_pattern == "fracture":
        severity_levels = cfg.experiment.severity_levels
        return _build_fracture_loaders(
            root=root,
            batch_size=batch_size,
            normalize=normalize,
            severity_levels=severity_levels,
            train_subset_size=train_subset,
            test_subset_size=test_subset,
        )
    if distortion_pattern == "thinning":
        severity_levels = cfg.experiment.severity_levels
        return _build_thinning_loaders(
            root=root,
            batch_size=batch_size,
            normalize=normalize,
            severity_levels=severity_levels,
            train_subset_size=train_subset,
            test_subset_size=test_subset,
        )
    if distortion_pattern == "mnist_uncertainty_decomp_blur":
        return _build_blur_loaders(
            root=root,
            batch_size=batch_size,
            normalize=normalize,
            severity_levels=cfg.experiment.severity_levels,
            train_subset_size=train_subset,
            test_subset_size=test_subset,
        )

    if distortion_pattern == "mnist_uncertainty_decomp_fracture":
        return _build_fracture_loaders(
            root=root,
            batch_size=batch_size,
            normalize=normalize,
            severity_levels=cfg.experiment.severity_levels,
            train_subset_size=train_subset,
            test_subset_size=test_subset,
        )
    else:
        raise ValueError(f"Unknown distortion pattern: {distortion_pattern}")
