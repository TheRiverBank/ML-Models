import matplotlib.pyplot as plt
import numpy as np
import model_evaluation as me


class Node:
    """ Node class. Holds a feature and a threshold on that feature.
        If it is a leaf node, the class label holds the predicted class. """
    def __init__(self, class_label, prob):
        self.class_label = class_label
        self.prob = prob
        self.thr = None
        self.feat = None
        self.left = None
        self.right = None


class Tree:
    """ Decision tree """
    def __init__(self, num_classes, cur_depth):
        self.num_classes = num_classes
        self.cur_depth = cur_depth
        self.max_depth = 4  # Current max depth if nothing else is specified
        self.min_impurity = 0.05
        self.root = None

    def fit(self, X, y):
        """ Build the tree """
        self.root = self.create_nodes(X, y)

    def predict(self, X, threshold):
        """
        For each datapoint, traverse the tree until a leaf is reached.
        The predicted class is the class label of the leaf node or
        a probability threshold combination.
        """
        preds = []
        for datapoint in X:
            node = self.root
            # Traverse tree until a leaf is reached
            while node.right is not None and node.left is not None:
                # Choose the correct child based on datapoint value in feature and
                # node feature threshold
                node = node.left if datapoint[node.feat] < node.thr else node.right
            # Use threshold for choosing class 0 or class 1
            preds.append(0 if node.prob >= threshold else 1)
        return preds

    def set_depth(self, max_depth):
        self.max_depth = max_depth

    def create_nodes(self, X, y, cur_depth = 0):
        """ Recursively generate new nodes """
        # Assign class to label by which class is most frequent
        prob = np.sum(y == 0) / y.size
        node_class_label = 1 if prob < 0.5 else 0
        # Create node
        node = Node(node_class_label, prob)

        # Recursively create new nodes
        if cur_depth < self.max_depth:
            # If impurity acceptable don't split
            if self.impurity(y) < self.min_impurity:
                node.left = node.right = None
            else:
                node = self.setup_child_nodes(X, y, node, cur_depth)

        return node

    def setup_child_nodes(self, X, y, parent_node, cur_depth):
        """ Takes a node and creates new child nodes by finding the best split """
        # Find best split based on instances that has reached this node
        threshold, feature = self.split(X, y)

        # Set feature and threshold for current node
        parent_node.feat = feature
        parent_node.thr = threshold

        # Get datapoints for branch 1
        X_0, y_0 = X[X[:, feature] < threshold], y[X[:, feature] < threshold]
        # Get datapoints for branch 2
        X_1, y_1 = X[X[:, feature] >= threshold], y[X[:, feature] >= threshold]

        # Create new nodes
        parent_node.left = self.create_nodes(X_0, y_0, cur_depth + 1)
        parent_node.right = self.create_nodes(X_1, y_1, cur_depth + 1)

        return parent_node

    def impurity(self, y):
        """ Measured impurity of node by entropy """
        # Check list of labels is not empty
        if (np.shape(y)[0]) == 0:
            return 0

        # Get posterior probability of class 0
        p = np.shape(y[y==0])[0] / np.shape(y)[0]

        # Avoid value error by checking for 0 and 1
        if p == 0:
            return (1 - p) * np.log2(1 - p)
        elif p == 1:
            return -p * np.log2(p)

        return (-p * np.log2(p)) - ((1 - p) * np.log2(1 - p))

    def total_impurity(self, Nm_1, Nm_2):
        """
        Total impurity by entropy.
        Nm_1 and Nm_2 are datapoints below and above threshold.
        """
        tot_impurity = 0
        # Total number of samples
        Nm = np.shape(Nm_1)[0] + np.shape(Nm_2)[0]

        # Calculate total impurity
        for N in (Nm_1, Nm_2):
            p_sum = 0
            N_len = np.shape(N)[0]

            for c in range(0, self.num_classes):
                p = np.shape(N[N == c])[0] / N_len
                # Add a small value to prevent log error
                p_sum += p * np.log2(p + 0.00000001)

            tot_impurity += -(N_len / Nm) * p_sum

        return tot_impurity

    def split(self, X, y):
        """
        Iterate over features and perform each possible split.
        The best split is the one that has the smallest total impurity.
        """
        threshold = 0
        feature = None
        min_ent = 1  # Max impurity of split

        for i in range(np.shape(X)[1]):
            # Get values in the feature
            feat_vals = X[:, i]

            # Iterate over values in feature and use each value as a threshold
            for t in feat_vals[:-1]:
                # Split on threshold (branch 1 and branch 2)
                Nm_1, Nm_2 = y[feat_vals <= t], y[feat_vals >= t]
                # Calculate total impurity
                tot_impurity = self.total_impurity(Nm_1, Nm_2)
                # If total impurity is less than current best split,
                # set new best attributes
                if tot_impurity < min_ent:
                    threshold = t
                    feature = i
                    min_ent = tot_impurity

        return threshold, feature

    def get_best_depth(self, start, end, step, model, X_train, y_train):
        """ Fits the model on depths from start to end with step size=step. """
        # Split training data
        X_train, y_train, X_test, y_test = me.validation_split(X_train, y_train)
        self.accuracies = []
        self.depths = []
        best_accuracy = 0
        best_depth = 0

        for depth in range(start, end + 1, step):
            self.depths.append(depth)
            print("Evaluating depth", depth)
            model.set_depth(depth)
            model.fit(X_train, y_train)

            preds = self.predict(X_test, threshold=0.5)
            accuracy = np.sum(preds == y_test) / y_test.size
            self.accuracies.append(accuracy)
            print("Accuracy:", accuracy, "\n")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_depth = depth

        return best_depth

    def plot_depth_accuracy(self):
        """
        Plots accuracies at each depth performed by get_best_depth()
        Must be called after get_best_depth()
        """
        plt.plot(self.depths, self.accuracies)
        plt.xlabel("Depth")
        plt.ylabel("Accuracy")
        plt.show()

def print_tree(node, depth=0):
    """ Prints out the tree to the console """
    if node is not None:
        if node.class_label == 0:
            c = "C0"
        else:
            c = "C1"

        print_tree(node.left, depth + 1)
        if node.thr is None:
            print(" " * 2 * depth + " ", c)
        else:
            print(" " * 2 * depth + " ", node.feat, "<", node.thr)
        print_tree(node.right, depth + 1)


if __name__ == '__main__':
    train_data = np.loadtxt("./data/seals_train.csv")
    test_data = np.loadtxt("./data/seals_test.csv")

    tree = Tree(num_classes=2, cur_depth=0)

    # Uncomment to find best depth, may take some time to complete
    # best_depth = tree.get_best_depth(1, 10, 1, tree, train_data[:, 1:], train_data[:, 0])
    # tree.plot_depth_accuracy()

    # Best depth on the validation set is found to be 3
    best_depth = 3
    tree.set_depth(best_depth)

    # Uncomment to find best features, may take some time to complete
    # best_feats = me.subset_selection(train_data[:, 1:], train_data[:, 0], tree)

    # Best features found by subset selection, add 1 since idx 0 is label
    best_feats = [0, 1]
    X_train, X_test = train_data[:, [x + 1 for x in best_feats]], \
                      test_data[:, [x + 1 for x in best_feats]]
    y_train, y_test = train_data[:, 0], test_data[:, 0]
    tree.fit(X_train, y_train)

    # Uncomment to print tree
    # node = tree.root
    # print_tree(node)

    preds = tree.predict(X_test, threshold=0.5)

    me.roc_and_auc(X_test, y_test, 0, tree)
    me.roc_and_auc(X_test, y_test, 1, tree)

    c0_mat = me.confusion_matrix(preds, y_test, 0)
    c1_mat = me.confusion_matrix(preds, y_test, 1)
    me.evaluate_model(preds, y_test, 0)
    me.evaluate_model(preds, y_test, 1)

    print(c0_mat)
    print(c1_mat)

    # Look at miss classifications
    #miss_class = me.plot_miss_classifications(y_test, np.array(preds))
