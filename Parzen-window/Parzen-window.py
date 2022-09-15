import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

def normal_kernel(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)


def parzen_windows_est(X, h):
    p = (1/(len(X) * h))
    results = [p * np.sum(normal_kernel((X[i] - X)/h)) for i in range(len(X))]

    return results


def test_random_data():
    n = 256
    h = 0.1
    X = np.random.uniform(0, 2, n)
    x = np.linspace(-1, 3, n)
    y = uniform.pdf(x, 0, 2)
    X = np.sort(X)
    est = parzen_windows_est(X, h)

    f, ax = plt.subplots(1)

    ax.plot(x, y)
    ax.plot(X, est)
    ax.set_ylim(ymin=0)
    plt.show()


if __name__ == '__main__':
    test_random_data()
