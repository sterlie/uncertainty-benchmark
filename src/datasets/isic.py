import os

import numpy as np
import PIL.Image as Image
import torch


class SkinISICDataset(torch.utils.data.Dataset):
    """Custom dataset for ISIC images."""

    def __init__(self, img_dir, groundtruth_csv_file, transform=None):
        super(SkinISICDataset, self).__init__()

        self.images = []
        self.labels = []

        for _, row in groundtruth_csv_file.iterrows():
            # Preferred path from adapter; fallback keeps original image_id-based behavior.
            if "image_path" in groundtruth_csv_file.columns:
                self.images.append(str(row["image_path"]))
            else:
                self.images.append(os.path.join(img_dir, str(row["image_id"]) + ".jpg"))
            self.labels.append(int(row["label"]))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image))
        return image.float(), torch.tensor(label, dtype=torch.long)
