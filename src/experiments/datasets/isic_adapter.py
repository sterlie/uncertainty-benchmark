from __future__ import annotations

import os
from typing import Dict

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.isic import SkinISICDataset
from src.experiments.datasets.base import DatasetExperimentAdapter, LoaderBundle


def _subset_df(df: pd.DataFrame, subset_size):
    if subset_size is None:
        return df
    n = min(int(subset_size), len(df))
    return df.iloc[:n].reset_index(drop=True)


def _build_transform(image_size: int, normalize: bool = True):
    tfms = [
        transforms.ToPILImage(),
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
    ]
    if normalize:
        tfms.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms.Compose(tfms)


def _derive_binary_label(merged_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    out = merged_df.copy()

    explicit_label_col = cfg.dataset.get("label_col", None)
    if explicit_label_col and str(explicit_label_col) in out.columns:
        out["label"] = out[str(explicit_label_col)].astype(int)
        return out

    malignant_classes = cfg.dataset.get(
        "malignant_classes",
        ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"],
    )
    malignant_classes = [str(c) for c in malignant_classes if str(c) in out.columns]

    if not malignant_classes:
        raise ValueError(
            "Unable to derive labels from groundtruth. "
            "Set dataset.label_col or dataset.malignant_classes in config/dataset/isic.yaml"
        )

    out["label"] = (out[malignant_classes].astype(float).sum(axis=1) > 0.5).astype(int)
    return out


def _read_merged_table(cfg: DictConfig) -> pd.DataFrame:
    data_root = str(cfg.dataset.get("data_root", cfg.data.root))
    metadata_csv = str(cfg.dataset.get("metadata_csv", os.path.join(data_root, "metadata.csv")))
    groundtruth_csv = str(cfg.dataset.get("groundtruth_csv", os.path.join(data_root, "groundtruth.csv")))

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"ISIC metadata not found: {metadata_csv}")
    if not os.path.exists(groundtruth_csv):
        raise FileNotFoundError(f"ISIC groundtruth not found: {groundtruth_csv}")

    meta = pd.read_csv(metadata_csv)
    gt = pd.read_csv(groundtruth_csv)

    lesion_col = str(cfg.dataset.get("lesion_id_col", "lesion_id"))
    if lesion_col not in meta.columns or lesion_col not in gt.columns:
        raise ValueError(
            f"lesion_id_col '{lesion_col}' must exist in both metadata and groundtruth"
        )

    merged = meta.merge(gt, on=lesion_col, how="inner")
    if len(merged) == 0:
        raise ValueError("ISIC metadata/groundtruth merge is empty. Check lesion_id values.")

    image_type_filter = cfg.dataset.get("image_type_filter", None)
    if image_type_filter and "image_type" in merged.columns:
        allowed = {str(x).strip().lower() for x in image_type_filter}
        merged = merged[merged["image_type"].astype(str).str.lower().isin(allowed)].reset_index(drop=True)

    image_id_col = str(cfg.dataset.get("image_id_col", "isic_id"))
    if image_id_col not in merged.columns:
        raise ValueError(f"image_id_col '{image_id_col}' not found in merged ISIC table")

    merged = _derive_binary_label(merged, cfg)
    merged["image_id"] = merged[image_id_col].astype(str)

    images_dir = str(cfg.dataset.get("images_dir", os.path.join(data_root, "images")))
    nested = bool(cfg.dataset.get("nested_image_folders", False))

    if nested:
        merged["image_path"] = merged.apply(
            lambda r: os.path.join(images_dir, str(r[lesion_col]), f"{r['image_id']}.jpg"),
            axis=1,
        )
    else:
        merged["image_path"] = merged["image_id"].map(lambda i: os.path.join(images_dir, f"{i}.jpg"))

    return merged


def _make_loader(df: pd.DataFrame, cfg: DictConfig, shuffle: bool) -> DataLoader:
    image_size = int(cfg.dataset.get("image_size", 224))
    normalize = bool(cfg.experiment.get("normalize", True))
    batch_size = int(cfg.dataset.batch_size)
    num_workers = int(cfg.dataset.get("num_workers", 0))
    images_dir = str(cfg.dataset.get("images_dir", os.path.join(str(cfg.dataset.get("data_root", cfg.data.root)), "images")))

    ds = SkinISICDataset(
        img_dir=images_dir,
        groundtruth_csv_file=df[["image_id", "image_path", "label"]].reset_index(drop=True),
        transform=_build_transform(image_size, normalize=normalize),
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

    train_df = _subset_df(train_df, train_subset)
    val_df = _subset_df(val_df, test_subset)
    test_df = _subset_df(test_df, test_subset)
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
    train_df = _subset_df(train_df, train_subset)
    val_df = _subset_df(val_df, test_subset)
    test_df = _subset_df(test_df, test_subset)
    return train_df, val_df, test_df


def build_isic_loaders(cfg: DictConfig, distortion_pattern: str = "plain") -> LoaderBundle:
    merged = _read_merged_table(cfg)

    cv_enabled = bool(cfg.dataset.get("cv", {}).get("enabled", False))
    if cv_enabled:
        train_df, val_df, test_df = _split_kfold(cfg, merged)
    else:
        train_df, val_df, test_df = _split_single(cfg, merged)

    train_loader = _make_loader(train_df, cfg, shuffle=True)
    val_loader = _make_loader(val_df, cfg, shuffle=False)

    # Keep runner contract identical: dict of eval loaders keyed by level names.
    level_name = str(distortion_pattern) if str(distortion_pattern) else "plain"
    eval_loaders: Dict[str, DataLoader] = {level_name: _make_loader(test_df, cfg, shuffle=False)}
    level_names = [level_name]

    return train_loader, val_loader, eval_loaders, level_names


class ISICExperimentAdapter(DatasetExperimentAdapter):
    """ISIC adapter supporting single split and optional stratified k-fold."""

    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        return build_isic_loaders(cfg, distortion_pattern=distortion_pattern)

    def supports_cross_validation(self) -> bool:
        return True
