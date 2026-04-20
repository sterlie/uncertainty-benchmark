from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.chexpert import (
    DEFAULT_ALL_CLASSES,
    DEFAULT_SELECTED_CLASSES,
    CheXpertDataset,
    build_chexpert_subgroup_slices,
    build_chexpert_train_transform,
    build_chexpert_val_transform,
    map_age_chexpert,
    prepare_chexpert_table,
)
from src.experiments.datasets.base import DatasetExperimentAdapter, LoaderBundle, subset_df


def _read_merged_table(cfg: DictConfig) -> pd.DataFrame:
    data_root = str(cfg.dataset.get("data_root", cfg.data.root))
    metadata_csv = str(cfg.dataset.get("metadata_csv", os.path.join(data_root, "valid.csv")))
    images_dir = str(cfg.dataset.get("images_dir", os.path.join(data_root, "valid")))
    image_id_col = str(cfg.dataset.get("image_id_col", "Path"))
    frontal_lateral_filter = cfg.dataset.get("frontal_lateral", "Frontal")
    all_classes = list(cfg.dataset.get("all_classes", DEFAULT_ALL_CLASSES))
    selected_classes = list(cfg.dataset.get("selected_classes", all_classes))

    return prepare_chexpert_table(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        image_id_col=image_id_col,
        frontal_lateral_filter=frontal_lateral_filter,
        all_classes=all_classes,
        selected_classes=selected_classes,
    )


def _make_loader(
    df: pd.DataFrame,
    cfg: DictConfig,
    shuffle: bool,
    train: bool = False,
    population_division: str = "by_gender",
) -> DataLoader:
    image_size = int(cfg.dataset.get("image_size", 224))
    batch_size = int(cfg.dataset.batch_size)
    num_workers = int(cfg.dataset.get("num_workers", 0))
    images_dir = str(cfg.dataset.get("images_dir",
                     os.path.join(str(cfg.dataset.get("data_root", cfg.data.root)), "valid")))
    crop = cfg.dataset.get("crop", None)
    blur = cfg.dataset.get("blur", None)
    pseudo_rgb = bool(cfg.dataset.get("pseudo_rgb", False))
    age_lower = int(cfg.dataset.get("age_lower", 50))
    age_upper = int(cfg.dataset.get("age_upper", 70))

    transform = (build_chexpert_train_transform(image_size, crop=crop, blur=blur) if train
                 else build_chexpert_val_transform(image_size, crop=crop))

    # Pass all columns the dataset class may need
    keep_cols = ["image_id", "image_path", "labels"]
    for c in ("sex", "age", "disease_count"):
        if c in df.columns:
            keep_cols.append(c)

    ds = CheXpertDataset(
        img_dir=images_dir,
        metadata_df=df[keep_cols].reset_index(drop=True),
        transform=transform,
        population_division=population_division,
        age_lower=age_lower,
        age_upper=age_upper,
        pseudo_rgb=pseudo_rgb,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Patient-level splitting (mirrors CheXpertDataResampleModule.set_split)
# ---------------------------------------------------------------------------

def _set_split(df, train_frac, val_frac, test_frac, rs):
    """Random split at the dataframe level: test first, then train/val from remainder."""
    test = df.sample(frac=test_frac, random_state=rs)
    train_val = df.drop(index=test.index)
    train = train_val.sample(frac=train_frac / (train_frac + val_frac), random_state=rs)
    val = train_val.drop(index=train.index)
    return train, val, test


def _split_by_patient(cfg: DictConfig, df: pd.DataFrame):
    """Split at patient level to avoid data leakage, matching the original logic.

    Steps:
      1. Get unique patients with their metadata.
      2. Split patients into train / val / test.
      3. Map back to full image-level dataframe.
    """
    rs = int(cfg.seed)
    perc_train = float(cfg.dataset.get("train_ratio", 0.6))
    perc_val = float(cfg.dataset.get("val_ratio", 0.2))
    perc_test = float(cfg.dataset.get("test_ratio", 0.2))
    train_subset = cfg.dataset.get("train_subset", None)
    test_subset = cfg.dataset.get("test_subset", None)

    # Build patient-level info
    patient_col = "patient_id"
    patient_info = df[[patient_col]].drop_duplicates().reset_index(drop=True)

    # Split patients
    patient_train, patient_val, patient_test = _set_split(
        patient_info, perc_train, perc_val, perc_test, rs
    )

    train_pids = set(patient_train[patient_col])
    val_pids = set(patient_val[patient_col])
    test_pids = set(patient_test[patient_col])

    train_df = df[df[patient_col].isin(train_pids)].reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_pids)].reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_pids)].reset_index(drop=True)

    train_df = subset_df(train_df, train_subset)
    val_df = subset_df(val_df, test_subset)
    test_df = subset_df(test_df, test_subset)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_chexpert_loaders(cfg: DictConfig, distortion_pattern: str = "plain") -> LoaderBundle:
    merged = _read_merged_table(cfg)

    train_df, val_df, test_df = _split_by_patient(cfg, merged)

    population_division = cfg.dataset.get("population_division", "by_gender")
    alternate_population_division = cfg.dataset.get(
        "alternate_population_division", population_division)

    train_loader = _make_loader(train_df, cfg, shuffle=True, train=True,
                                population_division=population_division)
    val_loader = _make_loader(val_df, cfg, shuffle=False,
                              population_division=alternate_population_division)

    subgroup_patterns = {"by_gender", "by_age", "by_disease_count"}
    pattern = str(distortion_pattern) if str(distortion_pattern) else "plain"

    age_lower = int(cfg.dataset.get("age_lower", 50))
    age_upper = int(cfg.dataset.get("age_upper", 70))

    if pattern in subgroup_patterns:
        subgroup_frames = build_chexpert_subgroup_slices(
            test_df, pattern, age_lower=age_lower, age_upper=age_upper)
        eval_loaders = {
            name: _make_loader(sub_df, cfg, shuffle=False,
                               population_division=alternate_population_division)
            for name, sub_df in subgroup_frames.items()
            if len(sub_df) > 0
        }
        if not eval_loaders:
            raise ValueError(f"All subgroup slices for '{pattern}' are empty.")
        level_names = list(eval_loaders.keys())
    else:
        eval_loaders = {pattern: _make_loader(test_df, cfg, shuffle=False,
                                              population_division=alternate_population_division)}
        level_names = [pattern]

    return train_loader, val_loader, eval_loaders, level_names


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class CHEXPERTExperimentAdapter(DatasetExperimentAdapter):
    """CheXpert adapter – patient-level split, no cross-validation."""

    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        return build_chexpert_loaders(cfg, distortion_pattern=distortion_pattern)

    def supports_cross_validation(self) -> bool:
        return False