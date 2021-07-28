import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.utils import save_image
from PIL import Image

model = models.vgg19(pretrained=True).features

print(model)