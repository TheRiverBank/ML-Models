import numpy as np

class Model:
    """
    Naive Bayes classifier. Assumes pre fitted parameters.
    """
    def __init__(self, prior_0, prior_1, beta, alpha: int, mean, var):
        self.prior_0 = prior_0
        self.prior_1 = prior_1
        self.beta = beta
        self.alpha = alpha
        self.mean = mean
        self.var = var

    def predict_single(self, x):
        """
        Final prediction is the distribution that produces the highest value.
        Predicts a single observation using naive Bayes.
        """
        predict_0 = self.gamma_density(x) + np.log(self.prior_0)
        predict_1 = self.gaussian_density(x) + np.log(self.prior_1)

        # Return the most probable class
        return np.argmax((predict_0, predict_1))

    def predict_multiple(self, X):
        """
        Predicts a set of observations.
        """
        result = []
        for observation in X:
            prediction = self.predict_single(observation)
            result.append(prediction)
        return np.array(result)

    def gamma_density(self, x):
        gamma_func = 1
        # Alpha is integer so calculate (alpha-1) factorial.
        for i in range(1, self.alpha):
            gamma_func *= i
        gamma_dist = 1 / ((self.beta ** self.alpha) * gamma_func) * \
                    (x ** (self.alpha - 1)) * (np.exp(-x / self.beta))
        return gamma_dist

    def gaussian_density(self, x):
        gaus_dist = 1 / (np.sqrt(2 * np.pi * self.var)) * \
                    np.exp(-(x - self.mean) ** 2 / (2 * self.var))

        return gaus_dist