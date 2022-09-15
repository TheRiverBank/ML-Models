import numpy as np


class KNN:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, X):
        preds = []
        for x in X:
            distances = []
            for x_t in self.x_train:
                distances.append(np.linalg.norm(x - x_t))

            y_cpy = y_train
            distances, y_cpy = zip(*sorted(zip(distances, y_cpy)))
            k_nearest_classes = y_cpy[:self.k]
            preds.append(max(set(k_nearest_classes), key=k_nearest_classes.count))

        return preds


if __name__ == '__main__':
    means_c1 = np.array([1, 1])
    means_c2 = np.array([1.5, 1.5])
    variance = 0.2

    cov = np.array([[variance, 0], [0, variance]])

    train_x1 = np.random.multivariate_normal(means_c1, cov, 50)
    train_x2 = np.random.multivariate_normal(means_c2, cov, 50)
    X_train = np.concatenate((train_x1, train_x2))
    y_train = np.zeros(100)
    y_train[50:] = 1

    X1 = np.random.multivariate_normal(means_c1, cov, 100)
    X2 = np.random.multivariate_normal(means_c2, cov, 100)
    X = np.concatenate((X1, X2))
    y = np.zeros(200)
    y[len(y)//2:] = 1

    model = KNN(3, X_train, y_train)
    predictions = model.predict(X)

    n = 0
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            n += 1

    print(n/len(predictions))
