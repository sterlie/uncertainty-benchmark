import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from torchvision import transforms


DEFAULT_ALL_CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
DEFAULT_SELECTED_CLASSES = ["BCC", "SCCKA", "AKIEC", "NV", "BKL", "MEL"]


def build_isic_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(int(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_isic_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(int(image_size), scale=(0.7, 1.0)),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def prepare_isic_table(
    metadata_csv: str,
    groundtruth_csv: str,
    images_dir: str,
    lesion_id_col: str = "lesion_id",
    image_id_col: str = "isic_id",
    image_type_filter: Iterable[str] = ("dermoscopic",),
    all_classes: Iterable[str] = DEFAULT_ALL_CLASSES,
    selected_classes: Iterable[str] = DEFAULT_SELECTED_CLASSES,
    nested_image_folders: bool = True,
) -> pd.DataFrame:
    """Build normalized ISIC dataframe with multiclass labels and resolved image paths."""
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"ISIC metadata csv not found: {metadata_csv}")
    if not os.path.exists(groundtruth_csv):
        raise FileNotFoundError(f"ISIC groundtruth csv not found: {groundtruth_csv}")

    meta = pd.read_csv(metadata_csv)
    gt = pd.read_csv(groundtruth_csv)

    if lesion_id_col not in meta.columns or lesion_id_col not in gt.columns:
        raise ValueError(
            f"lesion_id_col '{lesion_id_col}' must exist in both metadata and groundtruth"
        )

    merged = meta.merge(gt, on=lesion_id_col, how="inner")
    if len(merged) == 0:
        raise ValueError("ISIC metadata/groundtruth merge is empty. Check lesion_id values.")

    # filter data based on image_type
    if image_type_filter and "image_type" in merged.columns:
        allowed = {str(x).strip().lower() for x in image_type_filter}
        merged = merged[merged["image_type"].astype(str).str.lower().isin(allowed)].reset_index(drop=True)

    if image_id_col not in merged.columns:
        raise ValueError(f"image_id_col '{image_id_col}' not found in merged ISIC table")

    all_classes = [str(c) for c in all_classes]
    selected_classes = [str(c) for c in selected_classes]
    
    # check if all selected classes exists in meta data 
    missing = [c for c in all_classes if c not in merged.columns]
    if missing:
        raise ValueError(f"GroundTruth CSV is missing expected class columns: {missing}")
    
    # check if selected classes has i col in meta data 
    selected_present = [c for c in selected_classes if c in merged.columns]
    # if selected classes are missing in metadata reaise error
    if len(selected_present) != len(selected_classes):
        raise ValueError("Some selected_classes are not present in merged table.")
    
    # save groundtruth class label in '_class_name' row 
    # save groundtruth class label as a float32 multilabel vector
    label_matrix = merged[selected_present].values.astype("float32")
    merged["label"] = [label_matrix[i] for i in range(len(label_matrix))]
    merged["image_id"] = merged[image_id_col].astype(str)


    # handled if images are svaed in a nested dir
    if nested_image_folders:
        merged["image_path"] = merged.apply(
            lambda r: os.path.join(images_dir, str(r[lesion_id_col]), f"{r['image_id']}.jpg"),
            axis=1,
        )
    else:
        merged["image_path"] = merged["image_id"].map(lambda i: os.path.join(images_dir, f"{i}.jpg"))

    return merged


def build_isic_subgroup_slices(df: pd.DataFrame, subgroup: str) -> Dict[str, pd.DataFrame]:
    """Split an ISIC dataframe into subgroup-specific slices matching legacy thresholds."""
    if subgroup == "age":
        col = "age_approx"
        return {
            "under_30": df[df[col] <= 30],
            "35": df[df[col] == 35],
            "40": df[df[col] == 40],
            "45": df[df[col] == 45],
            "50": df[df[col] == 50],
            "55": df[df[col] == 55],
            "60": df[df[col] == 60],
            "65": df[df[col] == 65],
            "70": df[df[col] == 70],
            "75": df[df[col] == 75],
            "80": df[df[col] == 80],
            "85": df[df[col] == 85],
        }

    if subgroup == "skin_tone":
        return {f"tone_{tone}": df[df["skin_tone_class"] == tone] for tone in [1, 2, 3, 4, 5]}

    if subgroup == "hair":
        col = "MONET_hair"
        return {
            "level_1": df[df[col] < 0.2],
            "level_2": df[(df[col] >= 0.2) & (df["MONET_hair"] < 0.3)],
            "level_3": df[(df[col] >= 0.3) & (df["MONET_hair"] < 0.5)],
            "level_4": df[df[col] >= 0.5],
        }

    if subgroup == "drop":
        col = "MONET_gel_water_drop_fluid_dermoscopy_liquid"
        return {
            "level_1": df[df[col] < 0.3],
            "level_2": df[(df[col] >= 0.3) & (df[col] < 0.4)],
            "level_3": df[df[col] >= 0.4],
        }

    if subgroup == "ink":
        col = "MONET_skin_markings_pen_ink_purple_pen"
        return {
            "level_1": df[df[col] < 0.2],
            "level_2": df[(df[col] >= 0.2) & (df[col] < 0.4)],
            "level_3": df[(df[col] >= 0.4) & (df[col] < 0.6)],
            "level_4": df[df[col] >= 0.6],
        }

    raise ValueError(
        f"Unknown ISIC subgroup '{subgroup}'. Expected one of: age, skin_tone, hair, drop, ink."
    )


class SkinISICDataset(torch.utils.data.Dataset):
    """Custom dataset for ISIC images."""

    def __init__(self, img_dir, groundtruth_csv_file, transform=None):
        super(SkinISICDataset, self).__init__()

        self.images = []
        self.labels = []

        for _, row in groundtruth_csv_file.iterrows():
            if "image_path" in groundtruth_csv_file.columns:
                self.images.append(str(row["image_path"]))
            else:
                self.images.append(os.path.join(img_dir, str(row["image_id"]) + ".jpg"))
            # label is a scalar class index (argmax of the one-hot vector)
            self.labels.append(int(np.array(row["label"], dtype="float32").argmax()))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image))
        return image.float(), label
