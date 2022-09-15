import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
CLASS1 = -1
CLASS2 = 1


class LinearDiscriminantClassifier:
    def __init__(self, X, y, p, fittype="pocket"):
        assert len(X) == len(y)
        self.w = np.array([0, 0, 0], dtype=float)
        self.w_history = []
        self.X = X
        self.y = y
        self.p = p
        self.fit_type = fit_type

    def train(self):
        if self.fit_type == "pocket":
            self.pocket_fit()
        elif self.fit_type == "lms":
            self.lms_fit()
        elif self.fit_type == "sse":
            self.sse_fit()

    def pocket_fit(self):
        """
        Lines must be linearly separable or training will not converge
        """
        n_correct = 0
        converged = False
        while not converged:
            self.w_history.append(list(self.w))
            for i in range(len(self.X)):
                x = self.X[i, :]
                wx = self.w @ x
                if self.y[i] == CLASS1 and wx <= 0:
                    self.w += self.p * x
                elif self.y[i] == CLASS2 and wx >= 0:
                    self.w -= self.p * x
                else:
                    # If no update, count up num correct classifications
                    n_correct += 1
                    if n_correct == len(self.X):
                        converged = True
                        break
                    continue

                # If the classification was incorrect, reset correct count
                n_correct = 0

    def lms_fit(self, iterations=1000):
        for i in range(iterations):
            for j in range(len(self.X)):
                self.w = self.w + self.p * (self.X[j, :].dot(self.y[j] - (self.X[j, :].T.dot(self.w))))
                self.w_history.append(self.w)

    def sse_fit(self):
        self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))

    def predict(self, X):
        preds = []
        for x in X:
            if (self.w[:-1].T @ x) + self.w[-1] > 0:
                preds.append(0)
            else: preds.append(1)
        return preds


def plot(X, w, w_hist):
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1

    plt.scatter(x1[:, 0], x1[:, 1], c='b')
    plt.scatter(x2[:, 0], x2[:, 1], c='r')
    xx = np.linspace(min1, max1)

    # Plot some learning history
    lines = w_hist[:1000]
    for w_vec in lines:
        yy = (-(w_vec[0] * xx) - w_vec[2]) / w_vec[1]
        plt.plot(xx, yy, c='pink', alpha=0.5)

    yy = (-(w[0] * xx) - w[2]) / w[1]
    plt.plot(xx, yy, c='g')
    plt.show()


if __name__ == '__main__':
    means_c1 = np.array([1, 1])
    means_c2 = np.array([0, 0])
    variance = 0.2

    cov = np.array([[variance, 0], [0, variance]])

    x1 = np.random.multivariate_normal(means_c1, cov, 50)
    x2 = np.random.multivariate_normal(means_c2, cov, 50)

    x1 = x1[x1[:, 0] + x1[:, 1] > 1]
    x2 = x2[x2[:, 0] + x2[:, 1] < 1]
    y1 = np.ones(len(x1)) * -1
    y2 = np.ones(len(x2))

    X = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    # Add the ones vector
    X = np.c_[X, np.ones(len(X))]

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    coef = clf.coef_[0]
    model = LinearDiscriminantClassifier(X, y, 0.01, fittype="sse")
    model.train()

    plot(X, model.w, model.w_history)