import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([1, 2, 3])
my_tensor_2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
x = torch.empty(size=(3, 3))


def convert_tensor():
    tensor = torch.arange(4)
    print(tensor.shape)
    print(tensor)
    print(tensor.bool())

def init_tensor(x):
    x = torch.zeros(3, 3)
    print(x)

    x = torch.randn(3, 3)
    print(x)

    x = torch.eye(5, 5)  # I, eye (one-hot encoding)
    print(x)

    x = torch.arange(start=0, end=5, step=1)
    print(x)

    x = torch.linspace(start=0.1, end=1, steps=10)
    print(x)

    x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
    print(x)

    x = torch.empty(size=(1, 5)).uniform_(0, 1)
    print(x)

    x = torch.diag(torch.ones(3))
    print(x)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(my_tensor)
    print(my_tensor_2)
    print(my_tensor_2.dtype)
    print(my_tensor_2.device)
    print(my_tensor_2.shape)

    print(x)

    convert_tensor()
