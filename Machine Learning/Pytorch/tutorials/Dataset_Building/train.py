# Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from customDataset import CatsAndDogDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu0')

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1
load_model = True

# Load Data
dataset = CatsAndDogDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

x = torch.randn(64, 1, 28, 28)
print(model(x).shape)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Save model function
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


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