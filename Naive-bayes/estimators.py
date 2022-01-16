import numpy as np

def beta_estimate(X, alpha):
    """
    Beta estimate from Gamma distribution.
    X: training set
    y: labels
    alpha: alpha
    """
    n = X.shape[0]
    beta = (1 / (n * alpha)) * np.sum(X)

    return beta

def mean_estimate(X):
    """
    Mean estimate of Gaussian distribution.
    X: training set
    """
    n = X.shape[0]
    mean = (1 / n) * np.sum(X)

    return mean

def variance_estimate(X, mean):
    """
    Variance estimate of Gaussian distribution.
    X: training set
    mean: mean of training set
    """
    n = X.shape[0]
    variance = (1 / n) * np.sum((X - mean)**2)

    return variance