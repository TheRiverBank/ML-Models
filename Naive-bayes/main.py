import numpy as np
import estimators as est
import model_evaluation as me
from model import Model

if __name__ == '__main__':
    data_train = np.loadtxt('./data/optdigits-1d-train.csv')
    data_test = np.loadtxt('./data/optdigits-1d-test.csv')

    X_train, y_train = data_train[:, 1], data_train[:, 0]
    X_test = data_test

    # Subset class 0 and class 1, not including label
    X_train_0, X_train_1 = data_train[y_train == 0][:, 1], data_train[y_train == 1][:, 1]

    # Fitted parameters model
    beta = est.beta_estimate(X_train_0, alpha=9)
    mean = est.mean_estimate(X_train_1)
    variance = est.variance_estimate(X_train_1, mean)
    prior_0 = X_train_0.shape[0] / X_train.shape[0]
    prior_1 = X_train_1.shape[0] / X_train.shape[0]

    print("beta:", beta, "mean:", mean, "variance:", variance, "\n")
    print("Prior 0's:", prior_0, "1's:", prior_1, "\n")

    # Plot distributions with histogram
    me.plot_class_dist(X_train_0, X_train_1, beta, 9, variance, mean)

    # Create the model
    model = Model(prior_0, prior_1, beta, 9, mean, variance)
    # Predict training set
    predictions = model.predict_multiple(X_train)

    # Evaluate model for class 0 and 1
    cm0 = me.evaluate_model(predictions, y_train, 0)
    cm1 = me.evaluate_model(predictions, y_train, 1)
    print("Confusion matrices:\n", cm0, "\n", cm1, "\n")
