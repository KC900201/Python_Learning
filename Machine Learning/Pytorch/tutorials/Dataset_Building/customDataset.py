# URL for dataset: https://www.kaggle.com/c/dogs-vs-cats/data?select=sampleSubmission.csv
# Sample dataset: https://www.kaggle.com/dataset/c75fbba288ac0418f7786b16e713d2364a1a27936e63f4ec47502d73d6ef30ab

import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from skimage import io


class CatsAndDogDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  # 25000

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc([index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
