import numpy as np


class LinearRegression():
    def __init__(self):
        self.learning_rate = 1e-2
        self.total_iterations = 10000

    def yHat(self, X, w):  # yhat = w1x1 + .... + wNxN , camel case for variable name
        return np.dot(w.T, X)

    def loss(self, yhat, y):
        L = 1 / self.m * np.sum(np.power(yhat - y, 2))
        return L

    def gradient_descent(self, w, X, y, yHat):
        dldW = 2 / self.m * np.dot(X, (yHat - y).T)  # gradient value
        # print(dldW.shape)

        w = w - self.learning_rate * dldW
        return w

    def main(self, X, y):
        x1 = np.ones((1, X.shape[1]))
        x = np.append(X, x1, axis=0)

        self.m = X.shape[1]
        self.n = X.shape[0]

        w = np.zeros((self.n, 1))

        for it in range(self.total_iterations + 1):
            yHat = self.yHat(X, w)
            loss = self.loss(yHat, y)

            if it % 2000 == 0:
                print(f'Cost at iteration {it} is {loss}')

            w = self.gradient_descent(w, X, y, yHat)

        return w


if __name__ == '__main__':
    X = np.random.rand(1, 500)
    y = 3 * X + np.random.rand(1, 500) * 0.1

    regression = LinearRegression()
    w = regression.main(X, y)
