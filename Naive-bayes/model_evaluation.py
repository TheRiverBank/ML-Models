import numpy as np
import matplotlib.pyplot as plt

def plot_class_dist(X_0, X_1, beta, alpha: int, var, mean):
    bin_num = 100

    """ Gamma plot """
    count_0, bins_0, _ = plt.hist(X_0, bins=bin_num, edgecolor='black', density=True)
    gamma_func = 1
    # Alpha is integer so calculate (alpha-1) factorial.
    for i in range(1, alpha):
        gamma_func *= i
    # Create Gamma distribution using the bins from the histogram
    gamma_dist = 1 / ((beta**alpha) * gamma_func) * \
                 (bins_0**(alpha-1)) * (np.exp(-bins_0/beta))
    plt.plot(bins_0, gamma_dist, linewidth=2, color='r',
             label="\u03B2\u0302: 0.018  \u03B1: 9")

    """ Gaussian plot """
    count_1, bins_1, _ = plt.hist(X_1, bins=bin_num, edgecolor='black', density=True)
    # Create Gaussian distribution using the bins from the histogram
    gaus_dist = 1 / (np.sqrt(2 * np.pi * var)) *\
                np.exp(-(bins_1 - mean) ** 2 / (2 * var))
    plt.plot(bins_1, gaus_dist, linewidth=2, color='g',
             label="\u03BC\u0302: 0.826  \u03C3\u0302\u00b2: 0.007")

    plt.legend()
    plt.show()

def plot_test_data(X_test):
    """
    Plot of test data that produces a secret message.
    It splits the data into two classes where each split
    is thought to contain only correct data points belonging to that class.
    The threshold used is not necessarily the true threshold.
    """
    bin_num = 20
    # Split test set by class
    X_0 = X_test[X_test < 0.4]
    X_1 = X_test[X_test >= 0.4]

    """ Class 0 """
    count_0, bins_0, _ = plt.hist(X_0, bins=bin_num, edgecolor='black',
                                  density=True, color='tab:blue', label="Class 0")
    """ Class 1 """
    count_1, bins_1, _ = plt.hist(X_1, bins=bin_num, edgecolor='black',
                                  density=True, color='tab:orange', label="Class 1")

    plt.legend()
    plt.show()

def confusion_matrix(y_pred, y_actual, c):
    """
    Returns a dict containing confusion matrix.
    c: 0 or 1, for class 0 or 1
    """
    num_observations = y_pred.shape[0]
    c_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(num_observations):
        if y_actual[i] == y_pred[i] == c:
            c_matrix["tp"] += 1
        if y_pred[i] == c and y_actual[i] != y_pred[i]:
            c_matrix["fp"] += 1
        if y_actual[i] == y_pred[i] != c:
            c_matrix["tn"] += 1
        if y_pred[i] != c and y_actual[i] != y_pred[i]:
            c_matrix["fn"] += 1

    return c_matrix

def evaluate_model(y_pred, y_actual, c):
    c_matrix = confusion_matrix(y_pred, y_actual, c)
    accuracy = (c_matrix["tp"] + c_matrix["tn"]) / y_pred.shape[0]
    precision = c_matrix["tp"] / (c_matrix["tp"] + c_matrix["fp"])
    recall = c_matrix["tp"] / (c_matrix["tp"] + c_matrix["fn"])
    print("Evaluation of class", c)
    print("accuracy:", accuracy, "precision:", precision, "recall:", recall, "\n")

    return c_matrix
