import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn


# Methods for dealing with imbalanced datasets:
# 1. Oversampling
# 2. Class weighting
# 3.

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transforms=my_transforms)


def main():
    pass


if __name__ == "__main__":
    main()
