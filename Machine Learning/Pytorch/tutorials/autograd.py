# To train neural network using Pytorch auto differentiation engine (autograd)
import torch
import torchvision
import torchvision.datasets as datasets  # Has standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch import nn

# Define a pretrained network
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
dataset = datasets.MNIST(root="/data/", train=True, transform=transforms, download=True)
labels = torch.rand(1, 1000)

# Forward pass
prediction = model(data)
loss = (prediction - labels).sum()
print(loss)
print(f"Backward prop loss: {loss.backward()}")

# load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() # gradient descent

# Fine tuning (set autograd = false)
for param in model.parameters():
    param.requires_grad = False

# replace layer using model.fc
model.fc = nn.Linear(512, 18)

# create tensor with autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

print(a)
print(b)

Q = 3*a**3 - b**2
print(Q)

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(9*a**2 == a.grad)
print(-2*b == b.grad)