# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional
import torchvision


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)  # implement batch normalization

    def forward(self, x):
        return self.relu(self.batchNorm(self.conv(x)))


class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(conv_block(in_channels, red_3x3, kernel_size=1),
                                     conv_block(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(conv_block(in_channels, red_5x5, kernel_size=1),
                                     conv_block(red_5x5, out_5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     conv_block(in_channels, out_1x1, kernel_size=1))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


# Continue 7/13/2021
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()

    def forward(self, x):
        pass
