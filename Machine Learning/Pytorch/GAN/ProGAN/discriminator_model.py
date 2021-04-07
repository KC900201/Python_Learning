import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import generator_model
from generator_model import factors, ConvBlock, WSConv2d

class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1))

        # this is for 4 x 4 resolution
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)

    def fade_in(self):
        pass

    def minibatch_std(self, x):
        pass

    def forward(self, x):
        pass
