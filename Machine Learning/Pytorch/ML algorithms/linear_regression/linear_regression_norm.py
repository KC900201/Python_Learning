import numpy as np


# Let X be shape: (training_examples, features)
# Let y be shape: (training_examples, 1)
# Output, w be shape: (features, 1)

def linear_regression_normal_equation(X, y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W


if __name__ == '__main__':
    X = np.array([[0, 1, 2]]).T  # T = transpose
    y = np.array([-1, 0, 1])

    print(X.shape)
    print(y.shape)

    W = linear_regression_normal_equation(X, y)
    print(f'Linear regression of W: {W}')

    X1 = np.random.rand(5000, 1)
    y1 = 5 * X1 + np.random.rand(5000, 1) * 0.1

    W1 = linear_regression_normal_equation(X1, y1)
    print(W1.shape)
    print(f'Linear regression of W1: {W1}')
