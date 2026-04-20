"""Unified chest X-ray experiment adapter for CheXpert and NIH datasets."""

from __future__ import annotations

import os

import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.chest import (
    CHEXPERT_CLASSES,
    CHEXPERT_GENDER_MAP,
    NIH_CLASSES,
    NIH_GENDER_MAP,
    ChestXrayDataset,
    build_chest_subgroup_slices,
    build_chest_train_transform,
    build_chest_val_transform,
    prepare_chest_table,
    _extract_chexpert_patient_id,
    _resolve_chexpert_path,
    _resolve_nih_path,
)
from src.experiments.datasets.base import DatasetExperimentAdapter, LoaderBundle, subset_df


# ── Dataset-specific defaults ────────────────────────────────────────────

_DATASET_DEFAULTS = {
    "chexpert": {
        "all_classes": CHEXPERT_CLASSES,
        "gender_map": CHEXPERT_GENDER_MAP,
        "image_id_col": "Path",
        "patient_id_col": None,
        "sex_col": "Sex",
        "age_col": "Age",
        "frontal_lateral_col": "Frontal/Lateral",
        "frontal_lateral_filter": "Frontal",
        "path_resolver": _resolve_chexpert_path,
        "patient_id_extractor": _extract_chexpert_patient_id,
        "age_direction": "descending",
        "disease_count_threshold": 1,
    },
    "nih": {
        "all_classes": NIH_CLASSES,
        "gender_map": NIH_GENDER_MAP,
        "image_id_col": "Image Index",
        "patient_id_col": "Patient ID",
        "sex_col": "Patient Gender",
        "age_col": "Patient Age",
        "frontal_lateral_col": None,
        "frontal_lateral_filter": None,
        "path_resolver": _resolve_nih_path,
        "patient_id_extractor": None,
        "age_direction": "ascending",
        "disease_count_threshold": 1,
    },
}


def _get_defaults(cfg: DictConfig) -> dict:
    name = str(cfg.dataset.name).lower()
    return _DATASET_DEFAULTS.get(name, _DATASET_DEFAULTS["chexpert"])


# ── Table reading ────────────────────────────────────────────────────────


def _read_merged_table(cfg: DictConfig) -> pd.DataFrame:
    defaults = _get_defaults(cfg)
    data_root = str(cfg.dataset.get("data_root", cfg.data.root))
    metadata_csv = str(cfg.dataset.get("metadata_csv", os.path.join(data_root, "valid.csv")))
    images_dir = str(cfg.dataset.get("images_dir", data_root))
    image_id_col = str(cfg.dataset.get("image_id_col", defaults["image_id_col"]))
    all_classes = list(cfg.dataset.get("all_classes", defaults["all_classes"]))
    selected_classes = list(cfg.dataset.get("selected_classes", all_classes))

    frontal_lateral_col = cfg.dataset.get("frontal_lateral_col", defaults["frontal_lateral_col"])
    frontal_lateral_filter = cfg.dataset.get("frontal_lateral", defaults["frontal_lateral_filter"])
    patient_id_col = cfg.dataset.get("patient_id_col", defaults["patient_id_col"])
    sex_col = cfg.dataset.get("sex_col", defaults["sex_col"])
    age_col = cfg.dataset.get("age_col", defaults["age_col"])

    return prepare_chest_table(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        image_id_col=image_id_col,
        patient_id_col=patient_id_col,
        sex_col=sex_col,
        age_col=age_col,
        frontal_lateral_col=frontal_lateral_col,
        frontal_lateral_filter=frontal_lateral_filter,
        all_classes=all_classes,
        selected_classes=selected_classes,
        path_resolver=defaults["path_resolver"],
        patient_id_extractor=defaults["patient_id_extractor"],
    )


# ── Loader construction ──────────────────────────────────────────────────


def _make_loader(
    df: pd.DataFrame,
    cfg: DictConfig,
    shuffle: bool,
    train: bool = False,
    population_division: str = "by_gender",
) -> DataLoader:
    defaults = _get_defaults(cfg)
    image_size = int(cfg.dataset.get("image_size", 224))
    batch_size = int(cfg.dataset.batch_size)
    num_workers = int(cfg.dataset.get("num_workers", 0))
    images_dir = str(cfg.dataset.get("images_dir", str(cfg.dataset.get("data_root", cfg.data.root))))
    crop = cfg.dataset.get("crop", None)
    blur = cfg.dataset.get("blur", None)
    pseudo_rgb = bool(cfg.dataset.get("pseudo_rgb", False))
    age_lower = int(cfg.dataset.get("age_lower", 50))
    age_upper = int(cfg.dataset.get("age_upper", 70))
    age_direction = str(cfg.dataset.get("age_direction", defaults["age_direction"]))
    disease_count_threshold = int(cfg.dataset.get("disease_count_threshold", defaults["disease_count_threshold"]))
    gender_map = defaults["gender_map"]

    transform = (build_chest_train_transform(image_size, crop=crop, blur=blur) if train
                 else build_chest_val_transform(image_size, crop=crop))

    keep_cols = ["image_id", "image_path", "labels"]
    for c in ("sex", "age", "disease_count"):
        if c in df.columns:
            keep_cols.append(c)

    ds = ChestXrayDataset(
        img_dir=images_dir,
        metadata_df=df[keep_cols].reset_index(drop=True),
        transform=transform,
        population_division=population_division,
        gender_map=gender_map,
        age_lower=age_lower,
        age_upper=age_upper,
        age_direction=age_direction,
        disease_count_threshold=disease_count_threshold,
        pseudo_rgb=pseudo_rgb,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ── Patient-level splitting ──────────────────────────────────────────────


def _set_split(df, train_frac, val_frac, test_frac, rs):
    """Random split: test first, then train/val from remainder."""
    test = df.sample(frac=test_frac, random_state=rs)
    train_val = df.drop(index=test.index)
    train = train_val.sample(frac=train_frac / (train_frac + val_frac), random_state=rs)
    val = train_val.drop(index=train.index)
    return train, val, test


def _split_by_patient(cfg: DictConfig, df: pd.DataFrame):
    rs = int(cfg.seed)
    perc_train = float(cfg.dataset.get("train_ratio", 0.6))
    perc_val = float(cfg.dataset.get("val_ratio", 0.2))
    perc_test = float(cfg.dataset.get("test_ratio", 0.2))
    train_subset = cfg.dataset.get("train_subset", None)
    test_subset = cfg.dataset.get("test_subset", None)

    patient_col = "patient_id"
    patient_info = df[[patient_col]].drop_duplicates().reset_index(drop=True)

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


# ── Main entry point ─────────────────────────────────────────────────────


def build_chest_loaders(cfg: DictConfig, distortion_pattern: str = "plain") -> LoaderBundle:
    defaults = _get_defaults(cfg)
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
    age_direction = str(cfg.dataset.get("age_direction", defaults["age_direction"]))
    disease_count_threshold = int(cfg.dataset.get("disease_count_threshold", defaults["disease_count_threshold"]))

    if pattern in subgroup_patterns:
        subgroup_frames = build_chest_subgroup_slices(
            test_df, pattern,
            age_lower=age_lower, age_upper=age_upper,
            age_direction=age_direction,
            disease_count_threshold=disease_count_threshold,
        )
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


# ── Adapter class ────────────────────────────────────────────────────────


class ChestExperimentAdapter(DatasetExperimentAdapter):
    """Unified chest X-ray adapter (CheXpert, NIH) – patient-level split, no cross-validation."""

    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        return build_chest_loaders(cfg, distortion_pattern=distortion_pattern)

    def supports_cross_validation(self) -> bool:
        return False
