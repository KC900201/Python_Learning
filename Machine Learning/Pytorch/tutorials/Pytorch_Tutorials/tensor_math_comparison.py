import torch


def main():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    # Addition
    z = x + y
    z_2 = torch.add(x, y)  # same
    print(z)
    print(z_2)

    # Subtraction
    z = z - x
    print(z)

    # Division
    z = torch.true_divide(x, y)
    print(z)

    # inplace operations
    t = torch.zeros(3)
    t.add_(x)
    t += x
    print(t)

    # Exponentiation
    z = x.pow(2)
    print(z)
    z = x ** 2
    print(z)

    # Matrix multiplication
    x1 = torch.rand((2, 4))
    x2 = torch.rand((5, 3))
    x3 = torch.mm(x1, x2)
    x3 = x1.mm(x2)

    # matrix exponentiation
    matrix_exp = torch.rand(5, 5)
    print(matrix_exp.matrix_power(3))

    # dot product
    z = torch.dot(x, y)
    print(z)

    # Batch Matrix Multiplication


if __name__ == '__main__':
    main()
