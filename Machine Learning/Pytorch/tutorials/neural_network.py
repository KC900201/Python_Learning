# Define a neural network in Pytorch

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Defining a Neural Network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 3*3 image dim
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        # Second 2D convolutional layer,taking in 32 input layers
        # output 64 convolutional features, with square kernel size of 3
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6 x 6 from image dimension
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(120, 84)
        # Third fully connected layer
        self.fc3 = nn.Linear(84, 10)

    # use forward() to feed forward data,
    # pass data into computation graph (neural network)
    def forward(self, x):
        # Pass data through conv1
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Pass data through conv2
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Run max pooling over x
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # # Flatten x with start_dim = 1
        # x = torch.flatten(x, 1)
        # # Pass data through fc1
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # # Apply softmax to output
        # output = F.log_softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1: ]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.01)

if __name__ == "__main__":
    my_nn = Net()
    print(my_nn)
    print(len(list(my_nn.parameters())))

    params = list(my_nn.parameters())
    print(params[0].size)
    # Generate random data (a random 28x28 image)
    # random_data = torch.rand((1, 1, 28, 28))
    # result = my_nn(random_data)
    # print(result)

    input = torch.randn(1, 1, 32, 32)

    # Create optimizer
    optimizer = my_nn.optimizer();
    optimizer.zero_grad(); # zero gradient buffers

    out = my_nn(input)
    print(f"Output: {out}")
    # out_backward = out.backward(torch.randn(1, 10))
    # print(f"Output backward: {out_backward}")

    # Calculate loss function
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(f"Loss: {loss}")

    # Back propagation
    my_nn.zero_grad()
    print("Bias grad before backward: " + str(my_nn.conv1.bias.grad)) # Print grad before backward
    loss.backward()
    print("Bias grad after backward: " + str(my_nn.conv1.bias.grad)) # Print grad after backward