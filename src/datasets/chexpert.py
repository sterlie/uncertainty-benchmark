import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms


DEFAULT_ALL_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
]
DEFAULT_SELECTED_CLASSES = DEFAULT_ALL_CLASSES

GENDER_MAP = {"Female": 0, "Male": 1}


def map_age_chexpert(age, age_lower=50, age_upper=70):
    """Map age to group index: 0 = old (>= upper), 1 = middle (>= lower), 2 = young."""
    if age >= age_upper:
        return 0
    elif age >= age_lower:
        return 1
    return 2


def center_crop(img):
    """Crop tensor image to a centered square (minimum of H, W)."""
    _, y, x = img.shape
    crop_size = min(y, x)
    start_x = x // 2 - (crop_size // 2)
    start_y = y // 2 - (crop_size // 2)
    return img[:, start_y:start_y + crop_size, start_x:start_x + crop_size]


def build_chexpert_val_transform(image_size: int = 224, crop: int = None):
    """Validation / test transform matching the original ChexpertDatasetNew."""
    tfms = []
    if crop is not None:
        tfms.append(T.Resize([crop, crop]))
        tfms.append(T.CenterCrop([image_size, image_size]))
    else:
        tfms.append(T.Lambda(center_crop))
        tfms.append(T.Resize([image_size, image_size]))
    return T.Compose(tfms)


def build_chexpert_train_transform(image_size: int = 224, crop: int = None, blur: int = None):
    """Training transform matching the original ChexpertDatasetNew with augmentation."""
    tfms = []
    if crop is not None:
        tfms.append(T.Resize([crop, crop]))
        tfms.append(T.RandomCrop([image_size, image_size]))
    else:
        tfms.append(T.Lambda(center_crop))
        tfms.append(T.Resize([image_size, image_size]))
    tfms.append(T.RandomHorizontalFlip(p=0.5))
    tfms.append(T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5))
    if blur is not None:
        tfms.append(T.GaussianBlur(kernel_size=blur, sigma=(0.5, 2.0)))
    return T.Compose(tfms)


def _extract_patient_id(csv_path: str) -> str:
    """Extract patient id from CheXpert path, e.g. 'CheXpert-.../valid/patient64541/...' -> 'patient64541'."""
    parts = csv_path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("patient"):
            return part
    return csv_path


def prepare_chexpert_table(
    metadata_csv: str,
    images_dir: str,
    image_id_col: str = "Path",
    frontal_lateral_col: str = "Frontal/Lateral",
    frontal_lateral_filter: str = "Frontal",
    all_classes: Iterable[str] = None,
    selected_classes: Iterable[str] = None,
) -> pd.DataFrame:
    """Build normalized CheXpert dataframe with multilabel targets and resolved image paths.

    Labels are kept as float32 vectors preserving -1 (uncertain), 0 (negative), 1 (positive),
    matching the original CheXpertDataResampleModule behaviour. NaN is filled to 0.
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"CheXpert metadata csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    if image_id_col not in df.columns:
        raise ValueError(f"Column '{image_id_col}' not found in CheXpert table")

    if frontal_lateral_col not in df.columns:
        raise ValueError(f"Column '{frontal_lateral_col}' not found in CheXpert table")

    # Filter by Frontal/Lateral, default uses only frontal images
    if frontal_lateral_filter is not None:
        allowed = str(frontal_lateral_filter).strip().lower()
        df = df[df[frontal_lateral_col].astype(str).str.lower() == allowed].reset_index(drop=True)

    # Determine class columns
    if all_classes is None:
        all_classes = DEFAULT_ALL_CLASSES
    if selected_classes is None:
        selected_classes = all_classes

    missing = [c for c in selected_classes if c not in df.columns]
    if missing:
        raise ValueError(f"CheXpert CSV is missing class columns: {missing}")

    # Remap labels: 1 → 1, -1 → -1, everything else (0, NaN) → 0
    for col in selected_classes:
        df[col] = df[col].fillna(0.0).apply(lambda x: 1.0 if x == 1 else (-1.0 if x == -1 else 0.0))

    # Build per-row label vectors as float32 numpy arrays (preserving -1)
    class_matrix = df[list(selected_classes)].values.astype(np.float32)
    labels = [class_matrix[i] for i in range(len(class_matrix))]

    # Resolve image paths
    def _resolve_path(csv_path: str) -> str:
        parts = csv_path.replace("\\", "/")
        marker = "valid/"
        idx = parts.find(marker)
        if idx >= 0:
            relative = parts[idx + len(marker):]
        else:
            relative = os.path.basename(parts)
        return os.path.join(images_dir, relative)

    out = pd.DataFrame({
        "image_id": df[image_id_col].astype(str),
        "image_path": df[image_id_col].astype(str).map(_resolve_path),
        "patient_id": df[image_id_col].astype(str).map(_extract_patient_id),
        "labels": labels,
    })

    # Preserve metadata columns for population division / subgroup slicing
    if "Sex" in df.columns:
        out["sex"] = df["Sex"].values
    if "Age" in df.columns:
        out["age"] = df["Age"].values

    # Disease count (number of positive labels per sample)
    positive_counts = (class_matrix > 0).sum(axis=1)
    out["disease_count"] = positive_counts.astype(int)

    return out


def build_chexpert_subgroup_slices(df: pd.DataFrame, subgroup: str,
                                   age_lower: int = 50, age_upper: int = 70) -> Dict[str, pd.DataFrame]:
    """Split a CheXpert dataframe into subgroup-specific slices matching the original logic."""
    if subgroup == "by_gender":
        return {
            sex: df[df["sex"] == sex].reset_index(drop=True)
            for sex in df["sex"].dropna().unique()
        }

    if subgroup == "by_age":
        df = df.copy()
        df["_age_group"] = df["age"].apply(lambda a: map_age_chexpert(a, age_lower, age_upper))
        return {
            f"age_group_{g}": df[df["_age_group"] == g].drop(columns=["_age_group"]).reset_index(drop=True)
            for g in sorted(df["_age_group"].unique())
        }

    if subgroup == "by_disease_count":
        return {
            "low_disease": df[df["disease_count"] <= 1].reset_index(drop=True),
            "high_disease": df[df["disease_count"] > 1].reset_index(drop=True),
        }

    raise ValueError(
        f"Unknown CheXpert subgroup '{subgroup}'. "
        "Expected one of: by_gender, by_age, by_disease_count."
    )


class CheXpertDataset(torch.utils.data.Dataset):
    """Custom dataset for CheXpert chest X-ray images.

    Matches the original ChexpertDatasetNew behaviour:
    - Loads images as RGB via PIL
    - Converts to tensor via T.ToTensor()
    - Applies augment transforms (CenterCrop/Resize + optional augmentation)
    - Labels are binarized at __getitem__ time: (label > 0) → 1.0, else → 0.0
    - Optionally returns meta and fine_meta for population division
    """

    def __init__(
        self,
        img_dir,
        metadata_df,
        transform=None,
        population_division="by_gender",
        age_lower=50,
        age_upper=70,
        pseudo_rgb=False,
        negative_label=False,
    ):
        super(CheXpertDataset, self).__init__()

        self.population_division = population_division
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.pseudo_rgb = pseudo_rgb
        self.negative_label = negative_label
        self.return_meta = False
        self.return_fine_meta = False
        self.transform = transform

        self.samples = []
        for _, row in metadata_df.iterrows():
            img_path = str(row["image_path"]) if "image_path" in metadata_df.columns else os.path.join(img_dir, str(row["image_id"]))
            label = np.array(row["labels"], dtype=np.float32)

            # Compute meta / fine_meta matching original logic
            if population_division == "by_gender":
                meta = GENDER_MAP.get(row.get("sex", ""), 0)
                fine_meta = meta
            elif population_division == "by_age":
                age_val = int(row.get("age", 0))
                meta = map_age_chexpert(age_val, age_lower=age_lower, age_upper=age_upper)
                fine_meta = age_val
            elif population_division == "by_disease_count":
                meta = int(int(np.sum(label > 0)) > 2)
                fine_meta = int(np.sum(label > 0))
            else:
                meta = 0
                fine_meta = 0

            self.samples.append({
                "image_path": img_path,
                "label": label,
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
            label = torch.where(
                label > 0.0, torch.ones_like(label),
                torch.where(label < -0.5, -torch.ones_like(label), torch.zeros_like(label)))
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