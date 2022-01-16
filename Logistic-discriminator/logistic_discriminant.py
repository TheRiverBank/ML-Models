import numpy as np
import model_evaluation as me

class Model:
    def __init__(self, epochs, lf):
        """
        Logistic discrimination model
        :param epochs: Number of iterations
        :param lf: Learning factor
        """
        self.ephocs = epochs
        self.lf = lf

    def fit(self, X, y):
        """
        Training the logistic discriminator
        """
        # Create a weight vector of 0's, check if features are 1D or multi dim
        try:
            self.w = np.zeros(np.shape(X)[1])
        except IndexError:
            self.w = np.zeros(1)

        # Normalize input
        X = self.normalize(X)

        # Do gradient decent to find weights
        for i in range(self.ephocs):
            o = np.dot(X, self.w)
            s = self.sigmoid(o)
            w_i_diff = np.dot(y - s, X)
            self.w += self.lf * w_i_diff

    def predict(self, X, threshold):
        # Normalize inputs then predict the points using the sigmoid function
        X = self.normalize(X)
        # If only one feature in X, reshape to 2D
        if len(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))

        pred = self.sigmoid(np.dot(X, self.w))
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0

        return pred

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def normalize(self, X):
        """ Normalize inputs """
        return (X - np.mean(X)) / np.std(X)


if __name__ == '__main__':
    train_data = np.loadtxt("./data/seals_train.csv", dtype=np.float128)
    test_data = np.loadtxt("./data/seals_test.csv", dtype=np.float128)

    model = Model(epochs=100, lf=0.1)

    # Do subset selection to find best features
    # best_feats = me.subset_selection(
    #   train_data[:, 1:], train_data[:, 0],
    #   model
    # )
    # print("The best features are:", best_feats, "\n")

    # The best features found by subset selection
    best_feats = [0, 1, 5, 95]

    # Split dataset using only the best feature indices.
    # Add 1 since labels are at index 0.
    X_train, X_test = train_data[:, [x+1 for x in best_feats]], \
                      test_data[:, [x+1 for x in best_feats]]
    y_train, y_test = train_data[:, 0], test_data[:, 0]

    model.fit(X_train, y_train)

    preds = model.predict(X_test, threshold=0.5)

    me.roc_and_auc(X_test, y_test, 0, model)
    me.roc_and_auc(X_test, y_test, 1, model)

    c1_mat = me.confusion_matrix(preds, y_test, 0)
    c2_mat = me.confusion_matrix(preds, y_test, 1)

    me.evaluate_model(preds, y_test, 0)
    me.evaluate_model(preds, y_test, 1)

    print(c1_mat)
    print(c2_mat)

    # Look at miss classifications
    #miss_class = me.plot_miss_classifications(y_test, preds)
