import numpy as np
import os


class KNearestNeighbor():
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, num_loops=2):
        if num_loops == 2:
            distances = self.compute_distance_two_loops(X_test)
        elif num_loops == 1:
            distances = self.compute_distance_one_loop(X_test)
        else:
            distances = self.compute_distance_vectorized(X_test)
        return self.predict_labels(distances)

    def compute_distance_vectorized(self, X_test):
        """
        Can be tricky to understand this, we utilize heavy
        vecotorization as well as numpy broadcasting.
        Idea: if we have two vectors a, b (two examples)
        and for vectors we can compute (a-b)^2 = a^2 - 2a (dot) b + b^2
        expanding on this and doing so for every vector lends to the
        heavy vectorized formula for all examples at the same time.
        """
        X_test_squared = np.sum(X_test ** 2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train ** 2, axis=1, keepdims=True)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)

        # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)
        return np.sqrt(
            self.eps + X_test_squared - 2 * two_X_test_X_train + X_train_squared.T
        )

    def compute_distance_one_loop(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            distances[i, :] = np.sqrt(np.sum((self.X_train - X_test[i, :]) ** 2, axis=1))

        return distances

    def compute_distance_two_loops(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :]) ** 2))

        return distances

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred


if __name__ == '__main__':
    # X = np.loadtxt("data.txt", delimiter=',')
    # y = np.loadtxt("targets.txt")
    HOME_PATH = './knn/example_data/'
    X = np.loadtxt(os.path.join(HOME_PATH, 'data.txt'), delimiter=",")
    y = np.loadtxt(os.path.join(HOME_PATH, 'targets.txt'))

    # train = np.random.randn(10, 4)
    # test = np.random.randn(1, 4)
    # num_examples = train.shape[0]
    #
    # distance = np.sqrt(
    #     np.sum(test ** 2, axis=1, keepdims=True) + np.sum(train ** 2, axis=1, keepdims=True) - 2 * np.sum(test * train))

    KNN = KNearestNeighbor(k=3)
    KNN.train(X, y)
    # KNN.train(train, np.zeros(num_examples))
    # corr_distance = KNN.compute_distance_two_loops(test)
    # print(f'The difference is: {np.sum(np.sum((corr_distance - distance) ** 2))}')
    y_pred = KNN.predict(X, num_loops=1)

    print(f'Accuracy: {sum(y_pred == y) / y.shape[0]}')
