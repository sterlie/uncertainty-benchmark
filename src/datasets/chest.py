"""Unified chest X-ray dataset module for CheXpert, NIH, and VinDr-CXR.

All three datasets share the same multilabel disease classification structure
with identical transform pipelines. Differences (column names, age grouping
direction, disease labels, path resolution, presence of gender/age metadata)
are parameterised via config and dataset-specific defaults in chest_adapter.py.

VinDr-CXR note: has no gender or age metadata; only by_disease_count population
division is meaningful, and only ambiguity experiments are supported.
"""

import os
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torchvision.transforms as T

# CheXpert classes  

CHEXPERT_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
]

CHEXPERT_GENDER_MAP = {"Female": 0, "Male": 1}

# NIH classes

NIH_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]

NIH_GENDER_MAP = {"F": 0, "M": 1}


# VIN classes
_VIN_LABELS_FULL = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "No finding",  # index 14 - "No finding" – dropped from training labels
]

VIN_CLASSES = _VIN_LABELS_FULL[:-1]  # 14 disease classes used for training

# Age mapping
def map_age(age, age_lower=50, age_upper=70, direction="descending"):
    """Map age to group index (3 groups).
    0 = young (<lower), 1 = middle, 2 = old
    """
    #if direction == "descending":
    #    if age >= age_upper:
    #        return 0
    #    elif age >= age_lower:
    #        return 1
    #    return 2
    #else:  # ascending
    if age < age_lower:
        return 0
    elif age < age_upper:
        return 1
    return 2

    Transformation 

def center_crop(img):
    """Crop tensor image to a centered square (minimum of H, W)."""
    _, y, x = img.shape
    crop_size = min(y, x)
    start_x = x // 2 - (crop_size // 2)
    start_y = y // 2 - (crop_size // 2)
    return img[:, start_y:start_y + crop_size, start_x:start_x + crop_size]


def build_chest_transform(image_size: int = 224, crop: int = None, training: bool = False,  blur: int = None):
    tfms = []
    if crop is not None:
        tfms.append(T.Resize([crop, crop]))
        tfms.append(T.RandomCrop([image_size, image_size]))
    else:
        tfms.append(T.Lambda(center_crop))
        tfms.append(T.Resize([image_size, image_size]))
    if blur is not None:
        tfms.append(T.GaussianBlur(kernel_size=blur, sigma=(0.5, 2.0)))
    if training: 
        tfms.append(T.RandomHorizontalFlip(p=0.5))
        tfms.append(T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5))
    return T.Compose(tfms)


# Data table preperation 

def _extract_chexpert_patient_id(csv_path: str) -> str:
    """Extract patient id from CheXpert path, e.g. '.../valid/patient64541/...' -> 'patient64541'."""
    parts = csv_path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("patient"):
            return part
    return csv_path


def _resolve_chexpert_path(csv_path: str, images_dir: str) -> str:
    """Strip the CheXpert prefix up to 'valid/' and join with images_dir."""
    parts = csv_path.replace("\\", "/")
    marker = "valid/"
    idx = parts.find(marker)
    if idx >= 0:
        relative = parts[idx + len(marker):]
    else:
        relative = os.path.basename(parts)
    return os.path.join(images_dir, relative)


def _resolve_nih_path(csv_path: str, images_dir: str) -> str:
    """NIH paths are relative filenames – just join with images_dir."""
    return os.path.join(images_dir, csv_path)


def build_nih_path_resolver(images_dir: str) -> Callable[[str, str], str]:
    """Return a path resolver for the full NIH dataset layout:
        {images_dir}/images_NNN/images/{filename}.png

    Scans all images_* subdirs once and builds a filename -> absolute path
    lookup for O(1) resolution.
    """
    lookup: dict = {}
    try:
        for top in os.scandir(images_dir):
            if not top.is_dir():
                continue
            inner = os.path.join(top.path, "images")
            if not os.path.isdir(inner):
                continue
            try:
                for entry in os.scandir(inner):
                    if entry.is_file():
                        lookup[entry.name] = entry.path
            except OSError:
                pass
    except OSError:
        pass

    if lookup:
        def _resolver(csv_path: str, _images_dir: str) -> str:
            return lookup.get(os.path.basename(csv_path),
                              os.path.join(_images_dir, csv_path))
        return _resolver

    return _resolve_nih_path


def prepare_chest_table(
    metadata_csv: str,
    images_dir: str,
    image_id_col: str = "Path",
    patient_id_col: Optional[str] = None,
    sex_col: Optional[str] = "Sex",
    age_col: Optional[str] = "Age",
    frontal_lateral_col: Optional[str] = "Frontal/Lateral",
    frontal_lateral_filter: Optional[str] = "Frontal",
    all_classes: Iterable[str] = None,
    selected_classes: Iterable[str] = None,
    path_resolver: Callable[[str, str], str] = None,
    patient_id_extractor: Callable[[str], str] = None,
    df_preprocessor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Builds a chest X-ray dataframe.

    Works for CheXpert, NIH, and VinDr-CXR via the configurable column names
    and path / patient-id resolution callbacks.

    Returns a DataFrame with columns:
        image_id, image_path, patient_id, labels, [sex], [age], disease_count
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    if df_preprocessor is not None:
        df = df_preprocessor(df)

    if image_id_col not in df.columns:
        raise ValueError(f"Column '{image_id_col}' not found in CSV")

    # Optional frontal/lateral filtering (CheXpert only)
    if frontal_lateral_col and frontal_lateral_col in df.columns and frontal_lateral_filter:
        allowed = str(frontal_lateral_filter).strip().lower()
        df = df[df[frontal_lateral_col].astype(str).str.lower() == allowed].reset_index(drop=True)

    # Disease classes
    if all_classes is None:
        all_classes = CHEXPERT_CLASSES
    if selected_classes is None:
        selected_classes = all_classes

    missing = [c for c in selected_classes if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing class columns: {missing}")

    # Remap labels: 1 → 1, -1 → -1, everything else (0, NaN) → 0
    for col in selected_classes:
        df[col] = df[col].fillna(0.0).apply(lambda x: 1.0 if x == 1 else (-1.0 if x == -1 else 0.0))

    class_matrix = df[list(selected_classes)].values.astype(np.float32)
    labels = [class_matrix[i] for i in range(len(class_matrix))]

    # Resolve image paths
    if path_resolver is None:
        path_resolver = _resolve_chexpert_path
    image_paths = [path_resolver(p, images_dir) for p in df[image_id_col].astype(str)]

    # Resolve patient ids
    if patient_id_col and patient_id_col in df.columns:
        patient_ids = df[patient_id_col].astype(str).values
    elif patient_id_extractor is not None:
        patient_ids = [patient_id_extractor(p) for p in df[image_id_col].astype(str)]
    else:
        patient_ids = df[image_id_col].astype(str).values

    out = pd.DataFrame({
        "image_id": df[image_id_col].astype(str),
        "image_path": image_paths,
        "patient_id": patient_ids,
        "labels": labels,
    })

    # raw_labels: fractional (VinDr) or -1/0/1 (CheXpert) for ambiguity tasks.
    # Preprocessors can inject a '_raw_labels' column (list-per-row) before
    # binarisation; otherwise raw_labels == labels.
    if "_raw_labels" in df.columns:
        out["raw_labels"] = [np.array(r, dtype=np.float32) for r in df["_raw_labels"].tolist()]
    else:
        out["raw_labels"] = labels

    # Preserve metadata columns for population division / subgroup slicing
    if sex_col and sex_col in df.columns:
        out["sex"] = df[sex_col].values
    if age_col and age_col in df.columns:
        out["age"] = df[age_col].values

    positive_counts = (class_matrix > 0).sum(axis=1)
    out["disease_count"] = positive_counts.astype(int)

    return out


# Subgroup slicing

def build_chest_subgroup_slices(
    df: pd.DataFrame,
    subgroup: str,
    age_lower: int = 50,
    age_upper: int = 70,
    disease_count_threshold: int = 1,
    age_bin_size: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Split a chest X-ray dataframe into subgroup-specific slices."""
    if subgroup == "by_gender":
        return {
            sex: df[df["sex"] == sex].reset_index(drop=True)
            for sex in df["sex"].dropna().unique()
        }

    if subgroup == "by_age":
        df = df.copy()
        df["_age_group"] = df["age"].apply(
            lambda a: map_age(a, age_lower, age_upper))
        return {
            f"age_group_{g}": df[df["_age_group"] == g].drop(columns=["_age_group"]).reset_index(drop=True)
            for g in sorted(df["_age_group"].unique())
        }

    if subgroup == "by_age_fine":
        df = df.copy()
        slices = {}
        for bin_start in range(0, 100, age_bin_size):
            bin_end = bin_start + age_bin_size
            mask = (df["age"] >= bin_start) & (df["age"] < bin_end)
            if mask.any():
                slices[f"age_{bin_start}"] = df[mask].reset_index(drop=True)
        return slices

    if subgroup == "by_disease_count":
        return {
            "low_disease": df[df["disease_count"] <= disease_count_threshold].reset_index(drop=True),
            "high_disease": df[df["disease_count"] > disease_count_threshold].reset_index(drop=True),
        }

    raise ValueError(
        f"Unknown subgroup '{subgroup}'. "
        "Expected one of: by_gender, by_age, by_age_fine, by_disease_count."
    )


# Chest x-ray dataset class


class ChestXrayDataset(torch.utils.data.Dataset):
    """Unified dataset for chest X-ray images (CheXpert, NIH, VinDr-CXR).

    - Loads images as RGB via PIL
    - Converts to tensor via T.ToTensor()
    - Applies augment transforms (CenterCrop/Resize + optional augmentation)
    - Labels are binarized at __getitem__ time: (label > 0) → 1.0, else → 0.0
    - Optionally returns meta and fine_meta for population division
    - VinDr: no gender/age columns; use population_division='by_disease_count'
    """

    def __init__(
        self,
        img_dir,
        metadata_df,
        transform=None,
        population_division="by_gender",
        gender_map=None,
        age_lower=50,
        age_upper=70,        disease_count_threshold=1,
        pseudo_rgb=False,
        negative_label=False,
    ):
        super().__init__()

        self.population_division = population_division
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.disease_count_threshold = disease_count_threshold
        self.pseudo_rgb = pseudo_rgb
        self.negative_label = negative_label
        self.return_meta = False
        self.return_fine_meta = False
        self.transform = transform

        if gender_map is None:
            gender_map = CHEXPERT_GENDER_MAP

        has_raw = "raw_labels" in metadata_df.columns

        self.samples = []
        for _, row in metadata_df.iterrows():
            img_path = (str(row["image_path"]) if "image_path" in metadata_df.columns
                        else os.path.join(img_dir, str(row["image_id"])))
            label = np.array(row["labels"], dtype=np.float32)
            raw_label = np.array(row["raw_labels"], dtype=np.float32) if has_raw else label.copy()

            if population_division == "by_gender":
                meta = gender_map.get(row.get("sex", ""), 0)
                fine_meta = meta
            elif population_division == "by_age":
                age_val = int(row.get("age", 0))
                meta = map_age(age_val, age_lower=age_lower, age_upper=age_upper)
                fine_meta = age_val
            elif population_division == "by_disease_count":
                meta = int(int(np.sum(label > 0)) > disease_count_threshold)
                fine_meta = int(np.sum(label > 0))
            else:
                meta = 0
                fine_meta = 0

            self.samples.append({
                "image_path": img_path,
                "label": label,
                "raw_label": raw_label,
                "meta": meta,
                "fine_meta": fine_meta,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            raise IOError(f"Failed to load image: {sample['image_path']}")

        image = T.ToTensor()(image)

        label = torch.from_numpy(sample["label"])
        if self.negative_label:
            # unified ambiguity/disagreement threshold that works for both CheXpert and VinDr:
            #   raw == 1  (or >= 1)    → 1  (unanimous positive)
            #   raw == 0               → 0  (unanimous negative)
            #   raw < 0 OR 0 < raw < 1 → -1 (any inter-rater disagreement or CheXpert uncertain)
            raw = torch.from_numpy(sample["raw_label"])
            label = torch.where(
                raw >= 1.0, torch.ones_like(raw),
                torch.where(
                    (raw < 0) | (raw > 0),
                    -torch.ones_like(raw),
                    torch.zeros_like(raw),
                ),
            )
        else:
            label = (label > 0).float()

        if self.transform is not None:
            image = self.transform(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        if self.return_fine_meta:
            return image, label, torch.tensor([sample["meta"]]), torch.tensor([sample["fine_meta"]])
        if self.return_meta:
            return image, label, torch.tensor([sample["meta"]])
        return image, label
