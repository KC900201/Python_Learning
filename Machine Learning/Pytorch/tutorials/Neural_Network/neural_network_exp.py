# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as datasets


# Create Fully Connected
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784, 10)
x = torch.rand(64, 784)
print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu0')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST

# Initialize network

# Loss and optimizer

# Train Network

# Check accuracy on training & test to see how good our model
