import numpy as np
import matplotlib.pyplot as plt

def roc_and_auc(X_test, y_actual, class_num, model):
    """
    Classify test data using different sigmoid thresholds
    and monitor tp and fp-rate to plot ROC curve and print AUC value.
    """
    tp_lst = []
    fp_lst = []

    step = 0.05
    for threshold in np.arange(0, 1.0 + step, step):
        y_pred = model.predict(X_test, threshold)
        # Get the confusion matrix
        c_mat = confusion_matrix(y_pred, y_actual, class_num)
        # Add the tp-rate and fp-rate for this threshold
        tp_lst.append(c_mat["tp"] / (c_mat["tp"] + c_mat["fn"]))
        fp_lst.append(c_mat["fp"] / (c_mat["fp"] + c_mat["tn"]))

    # Plot the ROC
    plt.plot(fp_lst, tp_lst)
    plt.xlabel("fp-rate")
    plt.ylabel("tp-rate")
    plt.show()

    # Print the AUC
    print("Area under curve is", abs(np.trapz(tp_lst, fp_lst)), "\n")

def validation_split(X_train, y_train, size=0.1):
    """
    Splits data into 'size' sized validation set.
    The rest is used for traning.
    """
    X_split = int(np.shape(X_train)[0] * (1-size))
    y_split = int(np.shape(y_train)[0] * (1-size))

    X_train_ = X_train[:X_split, :]
    X_test_ = X_train[X_split:, :]
    y_train_ = y_train[:y_split]
    y_test_ = y_train[y_split:]

    return X_train_, y_train_, X_test_, y_test_

def subset_selection(X_train, y_train, model):
    """
    Do subset selection to find features that strengthens the model.
    Features are selected based on which has the lowest error on the validation set.
    """
    best_feats = []
    current_best_feat = 0
    current_best_feat_accuracy = 0

    # Create a list of the feature numbers
    feat_lst = list(range(np.shape(X_train)[1]))

    # Split the training data into train and validation
    X_train, y_train, X_test, y_test = validation_split(X_train, y_train)

    done = False
    while not done:
        found = 0
        for i in feat_lst:
            # Add new feature
            best_feats.append(i)
            # When X has only one feature, reshape to 2D
            if len(best_feats) == 1:
                X_train_new = np.reshape(X_train[:, best_feats], (-1, 1))
                X_test_new = np.reshape(X_test[:, best_feats], (-1, 1))
            else:
                X_train_new = X_train[:, best_feats]
                X_test_new = X_test[:, best_feats]

            model.fit(X_train_new, y_train)
            preds = model.predict(X_test_new, threshold=0.5)
            accuracy = np.sum(preds == y_test) / y_test.size

            # Check if feature made accuracy improve
            if accuracy > current_best_feat_accuracy:
                found = 1
                current_best_feat = i
                current_best_feat_accuracy = accuracy

            # Remove feature and try another one
            best_feats.remove(i)

        # End search if no new addition to the best features
        if found == 0:
            done = True

        # Add the best feature if it is not already in the list
        if current_best_feat not in best_feats:
            best_feats.append(current_best_feat)
            # Don't evaluate same feature again
            feat_lst.remove(current_best_feat)

    return best_feats

def confusion_matrix(preds, labels, c):
    """
    Creates a dictionary containing tp, fp, tn and fn.
    c: 0 or 1, for class 0 or 1
    """
    num_observations = np.shape(preds)[0]
    mtrx = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(num_observations):
        if labels[i] == preds[i] == c:
            mtrx["tp"] += 1
        if preds[i] == c and labels[i] != preds[i]:
            mtrx["fp"] += 1
        if labels[i] == preds[i] != c:
            mtrx["tn"] += 1
        if preds[i] != c and labels[i] != preds[i]:
            mtrx["fn"] += 1

    return mtrx

def evaluate_model(preds, labels, c):
    c_mat = confusion_matrix(preds, labels, c)
    accuracy = (c_mat["tp"] + c_mat["tn"]) / np.shape(preds)[0]
    try:
        precision = c_mat["tp"] / (c_mat["tp"] + c_mat["fp"])
    except ZeroDivisionError:
        precision = c_mat["tp"]
    try:
        recall = c_mat["tp"] / (c_mat["tp"] + c_mat["fn"])
    except ZeroDivisionError:
        recall = c_mat["tp"]

    print("Evaluation of class", c)
    print("accuracy:", accuracy, "precision:", precision, "recall:", recall, "\n")

    return c_mat

def plot_on_index(index):
    test_data = np.loadtxt("./seals_images_test.csv")
    img = test_data[index, :]
    img = np.reshape(img, (64, 64))
    plt.imshow(img, cmap="gray")
    plt.show()

def plot_miss_classifications(preds, labels):
    miss_class = np.argwhere(preds != labels)
    for i in miss_class:
        print("Should be", preds[i], "prediction is", labels[i])
        plot_on_index(i)
    return miss_class


