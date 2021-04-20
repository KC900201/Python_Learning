import torch


def main():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    # Addition
    z = x + y
    z_2 = torch.add(x, y)  # same
    # print(z)
    # print(z_2)

    # Subtraction
    z = z - x
    # print(z)

    # Division
    z = torch.true_divide(x, y)
    # print(z)

    # inplace operations
    t = torch.zeros(3)
    t.add_(x)
    t += x
    # print(t)

    # Exponentiation
    z = x.pow(2)
    # print(z)
    z = x ** 2
    # print(z)

    # Matrix multiplication
    x1 = torch.rand((2, 4))
    x2 = torch.rand((4, 2))
    x3 = torch.mm(x1, x2)
    # print(x3)
    # print(x3.shape)
    x3 = x1.mm(x2)
    # print(x3)

    # matrix exponentiation
    matrix_exp = torch.rand(5, 5)
    # print(matrix_exp.matrix_power(3))

    # dot product
    z = torch.dot(x, y)
    # print(z)

    # Batch Matrix Multiplication
    batch = 32
    n = 10
    m = 20
    p = 30

    tensor1 = torch.rand((batch, n, m))
    tensor2 = torch.rand((batch, m, p))
    out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)
    # print(tensor1)
    # print(tensor2)
    # print(out_bmm)

    # Example of Broadcasting
    x1 = torch.rand((5, 5))
    x2 = torch.rand((1, 5))
    print(x1)
    print(x2)

    z = x1 - x2
    print(z)
    z = x1 ** x2
    print(z)

#     Other useful tensor operations
    sum_x = torch.sum(x, dim=0)
    values, indices = torch.max(x, dim=0)
    values, indices = torch.min(x, dim=0)
    abs_x = torch.abs(x)

    z = torch.argmax(x, dim=0)
    z = torch.argmin(x, dim=0)

    mean_x = torch.mean(x.float(), dim=0)
    z = torch.eq(x, y)
    sorted_y, indices = torch.sort(y, dim=0, descending=False)

    z = torch.clamp(x, min=0)


if __name__ == '__main__':
    main()
