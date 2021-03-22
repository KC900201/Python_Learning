import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

    # sending x, y <- concatenate x,y along the channels

    class Discriminator(nn.Module):
        def __init__(self, in_channels=3, features=[64, 138, 256, 512]): # 256 -> 30x30
            super().__init__()
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2),
            )

            layers = []
            in_channels = features[0]
            for feature in features:
                return feature
#             Continue 3/22/2021
