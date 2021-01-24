"""
Training of DCGAN network on MNIST dataset with Discriminator and Generator import from models.py
"""
import torch
import torchvision
import torch.nn as nn # All neural networks, nn.Linear, nn.Conv2d
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc
import torchvision.datasets as datasets # Has standard datasets
import torchvision.transforms as transforms # Transformations we can perform on our dataset
from torch.utils.data import DataLoader # Gives easier dataset management
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
from models import Discriminator, Generator

# Continue 1/27/2021
