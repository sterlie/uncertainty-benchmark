from __future__ import annotations

import os
from typing import Dict

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.datasets.isic import (
    DEFAULT_ALL_CLASSES,
    DEFAULT_SELECTED_CLASSES,
    SkinISICDataset,
    build_isic_subgroup_slices,
    build_isic_train_transform,
    build_isic_val_transform,
    prepare_isic_table,
)
from src.experiments.datasets.base import DatasetExperimentAdapter, LoaderBundle, subset_df


def _read_merged_table(cfg: DictConfig) -> pd.DataFrame:
    data_root = str(cfg.dataset.get("data_root", cfg.data.root))
    metadata_csv = str(cfg.dataset.get("metadata_csv", os.path.join(data_root, "metadata.csv")))
    groundtruth_csv = str(cfg.dataset.get("groundtruth_csv", os.path.join(data_root, "groundtruth.csv")))
    lesion_col = str(cfg.dataset.get("lesion_id_col", "lesion_id"))
    image_id_col = str(cfg.dataset.get("image_id_col", "isic_id"))
    image_type_filter = cfg.dataset.get("image_type_filter", None)
    images_dir = str(cfg.dataset.get("images_dir", os.path.join(data_root, "images")))
    nested = bool(cfg.dataset.get("nested_image_folders", False))
    all_classes = cfg.dataset.get("all_classes", DEFAULT_ALL_CLASSES)
    selected_classes = cfg.dataset.get("selected_classes", cfg.dataset.get("six_classes", DEFAULT_SELECTED_CLASSES))

    return prepare_isic_table(
        metadata_csv=metadata_csv,
        groundtruth_csv=groundtruth_csv,
        images_dir=images_dir,
        lesion_id_col=lesion_col,
        image_id_col=image_id_col,
        image_type_filter=image_type_filter,
        all_classes=all_classes,
        selected_classes=selected_classes,
        nested_image_folders=nested,
    )


def _make_loader(df: pd.DataFrame, cfg: DictConfig, shuffle: bool, train: bool = False) -> DataLoader:
    image_size = int(cfg.dataset.get("image_size", 224))
    batch_size = int(cfg.dataset.batch_size)
    num_workers = int(cfg.dataset.get("num_workers", 0))
    images_dir = str(cfg.dataset.get("images_dir", os.path.join(str(cfg.dataset.get("data_root", cfg.data.root)), "images")))
    transform = build_isic_train_transform(image_size) if train else build_isic_val_transform(image_size)

    ds = SkinISICDataset(
        img_dir=images_dir,
        groundtruth_csv_file=df[["image_id", "image_path", "label"]].reset_index(drop=True),
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _split_single(cfg: DictConfig, df: pd.DataFrame):
    train_subset = cfg.dataset.get("train_subset", None)
    test_subset = cfg.dataset.get("test_subset", None)

    # Split by lesion to avoid leakage across two images per lesion.
    lesion_col = str(cfg.dataset.get("lesion_id_col", "lesion_id"))
    lesion_labels = (
        df[[lesion_col, "label"]]
        .drop_duplicates(subset=[lesion_col])
        .reset_index(drop=True)
    )

    train_lesions, test_lesions = train_test_split(
        lesion_labels,
        test_size=float(cfg.dataset.get("test_ratio", 0.2)),
        random_state=int(cfg.seed),
        stratify=lesion_labels["label"],
    )
    train_lesions, val_lesions = train_test_split(
        train_lesions,
        test_size=float(cfg.dataset.get("val_ratio", 0.2)),
        random_state=int(cfg.seed),
        stratify=train_lesions["label"],
    )

    train_df = df[df[lesion_col].isin(set(train_lesions[lesion_col]))].reset_index(drop=True)
    val_df = df[df[lesion_col].isin(set(val_lesions[lesion_col]))].reset_index(drop=True)
    test_df = df[df[lesion_col].isin(set(test_lesions[lesion_col]))].reset_index(drop=True)

    train_df = subset_df(train_df, train_subset)
    val_df = subset_df(val_df, test_subset)
    test_df = subset_df(test_df, test_subset)
    return train_df, val_df, test_df


def _split_kfold(cfg: DictConfig, df: pd.DataFrame):
    cv_cfg = cfg.dataset.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    fold_index = int(cv_cfg.get("fold_index", 0))
    seed = int(cv_cfg.get("seed", cfg.seed))

    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"dataset.cv.fold_index={fold_index} must be in [0, {n_splits - 1}]")

    lesion_col = str(cfg.dataset.get("lesion_id_col", "lesion_id"))
    lesion_labels = (
        df[[lesion_col, "label"]]
        .drop_duplicates(subset=[lesion_col])
        .reset_index(drop=True)
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(lesion_labels, lesion_labels["label"]))
    trainval_idx, test_idx = splits[fold_index]

    trainval_lesions = lesion_labels.iloc[trainval_idx].reset_index(drop=True)
    test_lesions = lesion_labels.iloc[test_idx].reset_index(drop=True)

    train_lesions, val_lesions = train_test_split(
        trainval_lesions,
        test_size=float(cfg.dataset.get("val_ratio", 0.2)),
        random_state=seed,
        stratify=trainval_lesions["label"],
    )

    train_df = df[df[lesion_col].isin(set(train_lesions[lesion_col]))].reset_index(drop=True)
    val_df = df[df[lesion_col].isin(set(val_lesions[lesion_col]))].reset_index(drop=True)
    test_df = df[df[lesion_col].isin(set(test_lesions[lesion_col]))].reset_index(drop=True)

    train_subset = cfg.dataset.get("train_subset", None)
    test_subset = cfg.dataset.get("test_subset", None)
    train_df = subset_df(train_df, train_subset)
    val_df = subset_df(val_df, test_subset)
    test_df = subset_df(test_df, test_subset)
    return train_df, val_df, test_df


def build_isic_loaders(cfg: DictConfig, distortion_pattern: str = "plain") -> LoaderBundle:
    merged = _read_merged_table(cfg)

    cv_enabled = bool(cfg.dataset.get("cv", {}).get("enabled", False))
    if cv_enabled:
        train_df, val_df, test_df = _split_kfold(cfg, merged)
    else:
        train_df, val_df, test_df = _split_single(cfg, merged)

    train_loader = _make_loader(train_df, cfg, shuffle=True, train=True)
    val_loader = _make_loader(val_df, cfg, shuffle=False)

    subgroup_patterns = {"age", "skin_tone", "hair", "drop", "ink"}
    pattern = str(distortion_pattern) if str(distortion_pattern) else "plain"

    if pattern in subgroup_patterns:
        subgroup_frames = build_isic_subgroup_slices(val_df, pattern)
        eval_loaders = {
            name: _make_loader(sub_df, cfg, shuffle=False)
            for name, sub_df in subgroup_frames.items()
            if len(sub_df) > 0
        }
        if not eval_loaders:
            raise ValueError(
                f"All subgroup slices for '{pattern}' are empty. "
                "Check that metadata includes the required subgroup columns."
            )
        level_names = list(eval_loaders.keys())
    else:
        eval_loaders = {pattern: _make_loader(test_df, cfg, shuffle=False)}
        level_names = [pattern]

    return train_loader, val_loader, eval_loaders, level_names


class ISICExperimentAdapter(DatasetExperimentAdapter):
    """ISIC adapter supporting single split and optional stratified k-fold."""

    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        return build_isic_loaders(cfg, distortion_pattern=distortion_pattern)

    def supports_cross_validation(self) -> bool:
        return True
