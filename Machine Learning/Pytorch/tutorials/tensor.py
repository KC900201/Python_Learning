import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device type: {device}\n")

data = [[1, 2], [3, 4]]

# default tensor data
x_data = torch.tensor(data)
print(x_data)
print(type(x_data))

# create tensor from Numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
print(type(x_np))

# Data manipulation
x_ones = torch.ones_like(x_data)
x_zeros = torch.zeros_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(f"Ones Tensor: \n{x_ones} \n")
print(f"Zeros Tensor: \n{x_zeros} \n")
print(f"Random Tensor: \n{x_rand} \n")

# Create tuple of tensor dimensions using shape()
shape = (2,3,) # row, col, dimension
rand_shape = torch.rand(shape)
ones_shape = torch.ones(shape)
zeros_shape = torch.zeros(shape)

print(f"Ones shape: \n{rand_shape} \n")
print(f"Zeros shape: \n{ones_shape} \n")
print(f"Random shape: \n{zeros_shape.to(device)} \n")

# Data attributes
rand_shape = rand_shape.to(device)
print(rand_shape.shape) # print shape
print(rand_shape.dtype) # print data type
print(rand_shape.device) # print device is stored to

# Joining tensors
t1 = torch.cat([ones_shape, zeros_shape], dim=1)
print(f"T1: \n{t1}\n")
print(f"T1: \n{t1.add_(5)}\n")

t_num = t1.numpy()
print(f"Value: {t_num}, Dtype: {type(t_num)}\n")