# Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu0')

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = False


# Modify pretrained model
class VGGMOD(nn.Module):
    def __init__(self):
        super(VGGMOD, self).__init__()

    def forward(self, x):
        return x


# Load a pretrain module from pytorch
model = torchvision.models.vgg16(pretrained=True)
model.avgpool = VGGMOD()  # apply VGGMOD
model.classifier = nn.Linear(512, 10)
model.to(device)

# print(model)

# Save model function
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Load Data
train_dataset = datasets.CIFAR10(root='../../../../data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='../../../../data/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
# model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train Network
for epoch in range(num_epochs):
    losses = []

    # Load model
    if epoch % 2 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    # print losses
    print(f'Loss  at epoch {epoch} :  {sum(losses) / len(losses):.5f}')


# Check accuracy
def check_acc(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%')

    model.train()

    # return acc


check_acc(train_loader, model)
check_acc(test_loader, model)
