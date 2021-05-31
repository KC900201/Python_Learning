# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard, include TensorBoard Writer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
num_epochs = 1
load_model = False


# Create CNN model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)  # FULLY connected layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


model = CNN()
x = torch.randn(64, 1, 28, 28)
print(model(x).shape)


# Save model function
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Load Data
train_dataset = datasets.MNIST(root='../../../../data/', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='../../../../data/', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#  Bad practice of setting up learning rates and batch sizes
batch_sizes = [2, 64, 1024]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Continue in doing hyperparameter search (5/24/2021)
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0

        # Initialize network and set to device (reset parameters by putting in loop)
        model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
        model.train()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate} test_tensorboard_writer')
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()

        # Train Network
        for epoch in range(num_epochs):
            losses = []
            accuracies = []

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

                # Calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                class_labels = [classes[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc1", model.fc1.weight)

                writer.add_scalar('Training loss', loss, global_step=step)
                writer.add_scalar('Training accuracy', running_train_acc, global_step=step)

                # if batch_idx == 230:
                #     writer.add_embedding(features, metadata=class_labels, label_img=data, global_step=batch_idx)
                step += 1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )

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

# check_acc(train_loader, model)
# check_acc(test_loader, model)
