import argparse
from mldata import parse_c45
import numpy as np
import os
from scipy.optimize import fmin_powell
import training
from utils import print_performance, timing, get_random_initial_weights, normalize

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '../data/')


class LogisticRegression(object):

    def __init__(self, c):
        self.c = c  # lambda term (python keyword)

        self._mean = 0
        self._std = 0
        self._optimal_weight_vector = None

    @classmethod
    @timing
    def solve(cls, data, c):
        return training.train_async(data, 5, cls, c)

    @staticmethod
    def _conditional_log_likelihood(example, label, weights):
        """
        Returns the likelihood term for an given example.
        :param example:
        :type example: ndarray
        :param label:
        :param weights:
        :return:
        """
        tmp = 1 + np.exp(-1 * np.dot(weights, example))
        likelihood_term = 0
        if label > 0:
            likelihood_term = np.log(1 / tmp)
        elif label < 0 and tmp != 1:
            likelihood_term = np.log(1 - (1 / tmp))
        return likelihood_term

    def _log_regression(self, weights, examples, labels):
        likelihood_sum = 0
        for example, label in zip(examples, labels):
            likelihood_sum += self._conditional_log_likelihood(example, label, weights)

        return -likelihood_sum + self.c * 0.5 * np.linalg.norm(weights, 2)**2  # from assignment prob. description

    def _minimize_log_regression_func(self, examples, labels, initial_weights):
        self._optimal_weight_vector = fmin_powell(
            func=self._log_regression,
            x0=initial_weights,
            args=(examples, labels),
            disp=False,
            xtol=0.1,
            ftol=0.1
        )

        self._optimal_weight_vector = np.array(self._optimal_weight_vector)

    def train(self, examples, labels, schema=None):
        self._mean = np.mean(examples, 0)
        self._std = np.std(examples, 0)
        initial_weights = get_random_initial_weights(examples)
        self._minimize_log_regression_func(examples, labels, initial_weights)

    def predict_example(self, example):
        """

        :param example:
        :type example: ndarray
        :return:
        """
        certainty = 1 / (1 + np.exp(-np.dot(self._optimal_weight_vector, example)))
        prediction = 1.0 if certainty > 0.5 else -1.0
        return prediction, certainty

    def classify(self, validation_set):
        return map(self.predict_example, validation_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Logistic Regression Classifier Implementation.")
    parser.add_argument('data_file_name')
    parser.add_argument('lambda_value', type=float)
    args = parser.parse_args()

    example_set = parse_c45(args.data_file_name, DATA_DIRECTORY)
    data_set = np.array(example_set.to_float())
    normalize(data_set, example_set.schema)
    results = LogisticRegression.solve(data_set, args.lambda_value)
    print_performance(results)
