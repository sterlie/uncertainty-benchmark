import torch
import pandas as pd
import os
import numpy as np
import PIL.Image as Image

from mpmath.identification import transforms


class SkinISICDataset(torch.utils.data.Dataset):
    '''
    custom dataset for ISIC dataset
    '''

    def __init__(self, img_dir, groundtruth_csv_file, transform=None):
        super(SkinISICDataset, self).__init__()

        self.images = []
        self.labels = []
        for i, row in groundtruth_csv_file.iterrows():
            self.images.append(os.path.join(img_dir, row['image_id']+'.jpg'))
            self.labels.append(row['label'])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label).long()