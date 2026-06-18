"""Unified chest X-ray experiment adapter for CheXpert, NIH, and VinDr-CXR datasets."""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.chest import (
    CHEXPERT_CLASSES,
    CHEXPERT_GENDER_MAP,
    NIH_CLASSES,
    NIH_GENDER_MAP,
    VIN_CLASSES,
    _VIN_LABELS_FULL,
    ChestXrayDataset,
    build_chest_subgroup_slices,
    build_chest_transform,
    build_chest_transform,
    map_age,
    prepare_chest_table,
    _extract_chexpert_patient_id,
    _resolve_chexpert_path,
    _resolve_nih_path,
    build_nih_path_resolver,
)
from src.experiments.datasets.base import DatasetExperimentAdapter, LoaderBundle, subset_df


def _resolve_vin_path(image_id: str, images_dir: str) -> str:
    """VinDr images live at {images_dir}/train_images/{image_id}.jpg."""
    return os.path.join(images_dir, "train_images", f"{image_id}.jpg")

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
        "disease_count_threshold": 1,
        "population_division": "by_gender",
        "df_preprocessor": None,
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
        "disease_count_threshold": 1,
        "population_division": "by_gender",
        "df_preprocessor": lambda df: _preprocess_nih_csv(df, NIH_CLASSES),
    },
    "vin": {
        "all_classes": VIN_CLASSES,
        "gender_map": {},  # VinDr-CXR has no gender metadata
        "image_id_col": "image_id",
        "patient_id_col": "image_id",  # each image is its own patient record
        "sex_col": None,
        "age_col": None,
        "frontal_lateral_col": None,
        "frontal_lateral_filter": None,
        "path_resolver": _resolve_vin_path,
        "patient_id_extractor": None,
        "disease_count_threshold": 2,  # >2 diseases = high burden
        "population_division": "by_disease_count",  # VinDr has no gender/age metadata
        "df_preprocessor": lambda df: _preprocess_vin_csv(df, VIN_CLASSES),
    },
}


def _get_defaults(cfg: DictConfig) -> dict:
    name = str(cfg.dataset.name).lower()
    return _DATASET_DEFAULTS.get(name, _DATASET_DEFAULTS["chexpert"])


# ── NIH-specific preprocessing ──────────────────────────────────────────


def _preprocess_nih_csv(df: pd.DataFrame, all_classes: list) -> pd.DataFrame:
    """Normalise a raw NIH Data_Entry-style CSV into the format expected by
    prepare_chest_table:

    1. Strip the trailing 'Y' from Patient Age (e.g. '060Y' → 60).
    2. Expand the pipe-separated 'Finding Labels' column into one binary
       column per disease class.
    """
    # --- Age ---
    if "Patient Age" in df.columns:
        df = df.copy()
        df["Patient Age"] = (
            df["Patient Age"]
            .astype(str)
            .str.rstrip("Yy")
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # --- Finding Labels → binary columns ---
    if "Finding Labels" in df.columns and all_classes:
        labels_series = df["Finding Labels"].astype(str)
        for cls in all_classes:
            if cls not in df.columns:
                df[cls] = labels_series.apply(
                    lambda s, c=cls: 1.0 if c in s.split("|") else 0.0
                )

    return df


def _preprocess_vin_csv(df: pd.DataFrame, all_classes: list) -> pd.DataFrame:
    """Aggregate annotation-level VinDr-CXR CSV into one row per image.

    The raw CSV has columns (image_id, rad_id, class_id) with one row per
    radiologist annotation.  This function:
      1. Groups by (image_id, rad_id) to collect each radiologist's class IDs.
      2. Builds a binary label vector per radiologist (15 classes).
      3. Averages across radiologists to get fractional labels in [0, 1].
      4. Binarises at >= 0.5 (majority agreement).
      5. Drops the "No finding" column.
    """
    df_by_rad = (
        df.groupby(["image_id", "rad_id"])["class_id"]
        .apply(lambda x: list(set(x.astype(int))))
        .reset_index(name="class_ids")
    )

    n_full = len(_VIN_LABELS_FULL)

    def _mean_vector(class_id_lists):
        vecs = np.zeros((len(class_id_lists), n_full), dtype=np.float32)
        for i, ids in enumerate(class_id_lists):
            for cid in ids:
                if 0 <= cid < n_full:
                    vecs[i, cid] = 1.0
        return np.mean(vecs, axis=0)

    df_by_image = (
        df_by_rad.groupby("image_id")["class_ids"]
        .apply(list)
        .reset_index(name="class_id_lists")
    )

    mean_vecs = np.stack(df_by_image["class_id_lists"].apply(_mean_vector).values)
    label_df = pd.DataFrame(mean_vecs, columns=_VIN_LABELS_FULL)
    # Store pre-binarised fractional values for the ambiguity task
    raw_label_cols = label_df.drop(columns=["No finding"])
    # Binarise at majority-agreement threshold
    label_df = (label_df >= 0.5).astype("float32")
    # Drop "No finding" – serves only as a negative marker
    label_df = label_df.drop(columns=["No finding"])

    result = pd.concat(
        [df_by_image["image_id"].reset_index(drop=True), label_df],
        axis=1,
    )
    # Attach raw fractional labels as a list column for prepare_chest_table to pick up
    result["_raw_labels"] = [
        raw_label_cols.iloc[i].values.astype("float32").tolist()
        for i in range(len(raw_label_cols))
    ]
    return result


# ── Table reading ────────────────────────────────────────────────────────


def _read_dataset_table(cfg: DictConfig) -> pd.DataFrame:
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

    # For NIH, build a resolver that handles both the flat layout and the
    # multi-subfolder layout (images_001 – images_012) used by the full dataset.
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name == "nih":
        path_resolver = build_nih_path_resolver(images_dir)
    else:
        path_resolver = defaults["path_resolver"]

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
        path_resolver=path_resolver,
        patient_id_extractor=defaults["patient_id_extractor"],
        df_preprocessor=defaults.get("df_preprocessor"),
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
    disease_count_threshold = int(cfg.dataset.get("disease_count_threshold", defaults["disease_count_threshold"]))
    gender_map = defaults["gender_map"]

    transform = build_chest_transform(image_size, crop=crop, blur=blur, training=train)
    
    #(build_chest_train_transform(image_size, crop=crop, training=True) if train
    #             else build_chest_val_transform(image_size, crop=crop))

    keep_cols = ["image_id", "image_path", "labels"]
    for c in ("sex", "age", "disease_count", "raw_labels"):
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


def _cap_per_patient(df: pd.DataFrame, num_per_patient: int, rs: int) -> pd.DataFrame:
    """Cap images per patient to *num_per_patient*, prioritising disease-positive rows.

    Replicates the old repo's prioritize_sampling: fill up to N with disease-positive
    images first, then pad with disease-negative ones if needed.
    """
    frames = []
    for _pid, group in df.groupby("patient_id"):
        n = min(num_per_patient, len(group))
        if n == len(group):
            frames.append(group)
            continue
        positive = group[group["disease_count"] > 0]
        negative = group[group["disease_count"] == 0]
        if len(positive) >= n:
            frames.append(positive.sample(n=n, random_state=rs))
        else:
            need = n - len(positive)
            neg_sample = negative.sample(n=min(need, len(negative)), random_state=rs)
            frames.append(pd.concat([positive, neg_sample]))
    return pd.concat(frames).reset_index(drop=True) if frames else df.iloc[:0].copy()


def _split_by_patient(cfg: DictConfig, df: pd.DataFrame):
    """Stratified patient-level splitting:

    1. Caps number of images per patient via *num_per_patient* (prioritize disease-positive).
    2. Assigns each patient a subgroup label based on *population_division*
       (by_gender / by_age / by_disease_count).
    3. Splits patients 60/20/20 *within* each (subgroup × has_disease) bucket
       and concatenates — preserving subgroup balance across splits.
    4. *train_subgroups*: optional list of subgroup values to keep in train/val
       (test always contains all subgroups, replicating female_perc_in_training=0
       / used_age_group behaviour from the old repo).
    """
    rs = int(cfg.seed) # set radom seed

    # set train/val/test ratio (default is 60/20/20)
    perc_train = float(cfg.dataset.get("train_ratio", 0.6))
    perc_val = float(cfg.dataset.get("val_ratio", 0.2))
    perc_test = float(cfg.dataset.get("test_ratio", 0.2))

    # sepcify if training/testing on subset or balansed data 
    train_subset = cfg.dataset.get("train_subset", None)
    test_subset = cfg.dataset.get("test_subset", None)

    # specify #image cap pr patient
    num_per_patient = cfg.dataset.get("num_per_patient", None)
    defaults = _get_defaults(cfg)

    _pop_div = cfg.dataset.get("population_division", None) or cfg.get("experiment", {}).get("population_division", None)
    if _pop_div is None:
        raise ValueError("'population_division' must be set in the experiment config (e.g. population_division: by_gender).")
    population_division = str(_pop_div)
    train_subgroups = cfg.dataset.get("train_subgroups", None)
    if train_subgroups is not None:
        train_subgroups = set(train_subgroups)

    age_lower = int(cfg.dataset.get("age_lower", 50))
    age_upper = int(cfg.dataset.get("age_upper", 70))
    disease_count_threshold = int(cfg.dataset.get("disease_count_threshold", defaults["disease_count_threshold"]))

    # 1. Per-patient image cap (disease-positive prioritised)
    if num_per_patient is not None:
        df = _cap_per_patient(df, int(num_per_patient), rs)

    # 2. Build one-row-per-patient info table with subgroup label
    def _patient_subgroup(group: pd.DataFrame):
        if population_division == "by_gender":
            return group["sex"].iloc[0] if "sex" in group.columns else "unknown"
        elif population_division == "by_age":
            age = int(group["age"].iloc[0]) if "age" in group.columns else 0
            return map_age(age, age_lower, age_upper)
        elif population_division == "by_disease_count":
            dc = int(group["disease_count"].iloc[0]) if "disease_count" in group.columns else 0
            return int(dc > disease_count_threshold)
        return "all"

    # build one-row-pr-patient table
    patient_rows = []
    for pid, group in df.groupby("patient_id"):
        sg = _patient_subgroup(group)
        has_disease = bool((group["disease_count"] > 0).any()) if "disease_count" in group.columns else False
        patient_rows.append({"pid": pid, "subgroup": sg, "has_disease": has_disease})
    patient_info_df = pd.DataFrame(patient_rows)

    # 3. Stratified split within each (subgroup × has_disease) bucket
    train_pids: set = set()
    val_pids: set = set()
    test_pids: set = set()

    # build train/val/test 
    for (_sg, _hd), bucket in patient_info_df.groupby(["subgroup", "has_disease"]):
        if len(bucket) == 0:
            continue
        b_train, b_val, b_test = _set_split(bucket[["pid"]], perc_train, perc_val, perc_test, rs)
        # Test always gets all subgroups
        test_pids.update(b_test["pid"])
        # Train/val respect train_subgroups filter
        if train_subgroups is None or _sg in train_subgroups:
            train_pids.update(b_train["pid"])
            val_pids.update(b_val["pid"])

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pids)].reset_index(drop=True)

    train_df = subset_df(train_df, train_subset)
    val_df = subset_df(val_df, test_subset)
    test_df = subset_df(test_df, test_subset)
    return train_df, val_df, test_df


# ── Main entry point ─────────────────────────────────────────────────────

def build_chest_loaders(cfg: DictConfig, distortion_pattern: str = "plain") -> LoaderBundle:
    defaults = _get_defaults(cfg)
    merged = _read_dataset_table(cfg)

    train_df, val_df, test_df = _split_by_patient(cfg, merged)

    _pop_div = cfg.dataset.get("population_division", None) or cfg.get("experiment", {}).get("population_division", None)
    if _pop_div is None:
        raise ValueError("Specify how to stratify train/val/test split" \
                        "'population_division' must be set in the experiment config (e.g. population_division: by_gender).")
    population_division = str(_pop_div)
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
    disease_count_threshold = int(cfg.dataset.get("disease_count_threshold", defaults["disease_count_threshold"]))

    if pattern in subgroup_patterns:
        subgroup_frames = build_chest_subgroup_slices(
            test_df, pattern,
            age_lower=age_lower, age_upper=age_upper,
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
    """Unified chest X-ray adapter (CheXpert, NIH, VinDr-CXR) – patient-level split, no cross-validation."""

    def build_loaders(self, cfg: DictConfig, distortion_pattern: str) -> LoaderBundle:
        return build_chest_loaders(cfg, distortion_pattern=distortion_pattern)

    def supports_cross_validation(self) -> bool:
        return False
