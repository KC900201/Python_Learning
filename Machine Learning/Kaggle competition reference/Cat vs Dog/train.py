# Import
# efficient net_pytorch -> https://github.com/lukemelas/EfficientNet-PyTorch
# anaconda install -> https://anaconda.org/conda-forge/efficientnet-pytorch
# Continue 11/12/2021

import os
import torch
import torch.nn.functional as F
import numpy as np
import config
import tqdm

from torch import nn, optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

from dataset import CatDog


def save_feature_vectors(model, loader, output_size=(1, 1), file="trainb7"):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_max_pool2d(features, output_size=output_size)

        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"data_features/X_{file}.npy", np.concatenate(images, axis=0))
    np.save(f"data_features/y_{file}.npy", np.concatenate(labels, axis=0))
    model.train()


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    train_dataset = CatDog(root="data/train/", transform=config.basic_transform)
    test_dataset = CatDog(root="data/test/", transform=config.basic_transform)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = model.to(config.DEVICE)

    save_feature_vectors(model, train_loader, output_size=(1, 1), file="train_b7")
    save_feature_vectors(model, test_loader, output_size=(1, 1), file="test_b7")


if __name__ == "__main__":
    main()
