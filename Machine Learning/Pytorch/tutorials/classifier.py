# How to train an image classifier
# Libraries
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from neural_network import Net

# Steps
# 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
# 2. Define a Convolutional Neural Network
# 3. Define a loss function
# 4. Train the network on the training data
# 5. Test the network on the test data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# path for saving trained model
PATH = './cifar_net_pth'

if __name__ == '__main__':
    if not (os.path.isfile(PATH)):
        # get random training images
        dataiter = iter(trainLoader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

        # Import neural network
        net = Net()
        # Train on GPU
        print(device)
        net.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Train network
        for epoch in range(5):
            running_loss = 0.0

            for i, data in enumerate(trainLoader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                # Continue here (02/22/2021)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished training')

        # Save our trained model
        torch.save(net.state_dict(), PATH)

    # test trained network on test data
    print('Now testing our trained model')
    dataiter = iter(testLoader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s ' % classes[labels[j]] for j in range(4)))

    # Load back our saved model and test see neural network
    net_2 = Net()
    net_2.load_state_dict(torch.load(PATH))
    outputs = net_2(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s ' % classes[predicted[j]] for j in range(4)))

    # Test on whole dataset
    print('Test on whole dataset\n')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            net_2.to(device)
            outputs = net_2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net_2(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # print(device)
