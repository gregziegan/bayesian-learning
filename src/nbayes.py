import argparse
from math import exp, log, pi
from mldata import parse_c45
import numpy as np
import os
import training
from utils import timing, print_performance, normalize

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '../data/')


class NaiveBayes(object):

    def __init__(self, m_estimate):
        self.m_estimate = m_estimate

        self._prob_pos_y = 0.0
        self._prob_neg_y = 0.0
        self._feature_probabilities = []  # will contain 2-tuples, each with the probabilities of y|xi and -y|xi

    @classmethod
    @timing
    def solve(cls, data, schema, m_estimate):
        return training.train_async(data, 5, cls, m_estimate, schema=schema)

    def train(self, examples, class_labels, schema):
        """
        Populates the list of feature probabilities.
        :param examples: training data, an array of feature arrays
        :type examples: ndarray
        :param class_labels: true labels associated with training data (1.0 or -1.0)
        :type class_labels: ndarray
        :param schema: object that contains meta information like the feature's type
        :type schema: Schema
        :return:
        """
        num_pos_labels = 0
        for label in class_labels:
            num_pos_labels += 1 if label > 0 else 0

        total_labels = float(len(class_labels))

        self._prob_pos_y = num_pos_labels / total_labels
        self._prob_neg_y = 1 - self._prob_pos_y

        for feature_index, feature in enumerate(schema):
            feature_set = [example[feature_index] for example in examples]

            if feature.type == 'CONTINUOUS':
                feature_prob_summary = self.train_continuous_feature(feature_set, class_labels)
            else:
                feature_prob_summary = self.train_nominal_feature(feature_set, class_labels, feature.values)

            self._feature_probabilities.append(feature_prob_summary)

    def classify(self, validation_data):
        """
        Returns a list of predictions on unlabeled data.
        :type validation_data: ndarray
        :return:
        """
        return map(self.predict_example, validation_data)

    def predict_example(self, example):
        """
        Predicts a new example using the probabilities stored in this instance's feature probabilities list.
        :param example: feature array
        :type example: ndarray
        :return:
        """
        pos_prob = log(self._prob_pos_y)
        neg_prob = log(self._prob_neg_y)

        for feature_index, feature_value in enumerate(example):
            get_conditional_prob_func = self._feature_probabilities[feature_index]['get_conditional_prob']
            conditional_prob_pos, conditional_prob_neg = get_conditional_prob_func(feature_value)

            if conditional_prob_pos > 0:
                pos_prob += log(conditional_prob_pos)

            if conditional_prob_neg > 0:
                neg_prob += log(conditional_prob_neg)

        estimate = 1.0 if pos_prob > neg_prob else -1.0
        certainty = pos_prob

        return estimate, certainty

    def train_nominal_feature(self, feature_set, labels, nominal_values):
        """
        Returns the probabilities of a class label associated with examples of a nominal feature.
        :param feature_set: contains all the values in the training set for a particular feature
        :param labels: an array of class labels (1.0 or -1.0)
        :param nominal_values:
        :return: a dictionary mapping nominal value keys to values containing the positive/negative label counts
        and a conditional probability function
        :rtype: dict
        """
        pos_bin = {v: 0 for v in nominal_values}
        neg_bin = {v: 0 for v in nominal_values}

        num_pos_class, num_neg_class = 0.0, 0.0

        for feature_value, label in zip(feature_set, labels):
            if label > 0:
                pos_bin[feature_value] += 1
                num_pos_class += 1.0
            else:
                neg_bin[feature_value] += 1
                num_neg_class += 1.0

        v = len(nominal_values)
        m = self.get_smoothing_estimate(v)
        prior = 1 / float(v)

        for nominal_value in nominal_values:
            pos_bin[nominal_value] = (pos_bin[nominal_value] + m * prior) / (num_pos_class + m)
            neg_bin[nominal_value] = (neg_bin[nominal_value] + m * prior) / (num_neg_class + m)

        summary = {
            'pos_bin': pos_bin,
            'neg_bin': neg_bin,
            'get_conditional_prob': lambda example: (pos_bin[example], neg_bin[example])
        }

        return summary

    def train_continuous_feature(self, feature_set, labels):
        """
        Returns the probabilities/statistics of a class label associated with examples of a continuous feature.
        :param feature_set: contains all the values in the training set for a particular feature
        :param labels: an array of class labels (1.0 or -1.0)
        :return: a dictionary containing a conditional probability function closure and various statistical information.
        :rtype: dict
        """
        positive_bin, negative_bin = [], []

        num_pos_class, num_neg_class = 0.0, 0.0

        for feature_example, label in zip(feature_set, labels):
            if label > 0:
                positive_bin.append(feature_example)
                num_pos_class += 1.0
            else:
                negative_bin.append(feature_example)
                num_neg_class += 1.0

        pos_mean = sum(positive_bin) / num_pos_class
        neg_mean = sum(negative_bin) / num_neg_class

        pos_variance = sum([(v - pos_mean)**2 for v in positive_bin]) / num_pos_class
        neg_variance = sum([(v - neg_mean)**2 for v in negative_bin]) / num_neg_class

        pos_variance = 0.01 if pos_variance < 0.01 else pos_variance
        neg_variance = 0.01 if neg_variance < 0.01 else neg_variance

        summary = locals()  # places all variables in this scope into a dictionary
        summary['get_conditional_prob'] = lambda e: self.get_continuous_conditional_probability(e, summary)
        return summary

    @staticmethod
    def get_continuous_conditional_probability(feature_value, summary):
        """
        Returns positive and negative probabilities using a gaussian distribution
        :param feature_value:
        :param summary: dictionary containing positive and negative class label mean and variance
        :return: tuple of positive and negative probabilities
        :rtype: tuple
        """
        pos_mu, neg_mu = summary['pos_mean'], summary['neg_mean']
        pos_sig2, neg_sig2 = summary['pos_variance'], summary['neg_variance']
        prob_pos = 1 / (2 * pi * pos_sig2)**0.5 * exp(-0.5 * (feature_value - pos_mu)**2 / pos_sig2)
        prob_neg = 1 / (2 * pi * neg_sig2)**0.5 * exp(-0.5 * (feature_value - neg_mu)**2 / neg_sig2)
        return prob_pos, prob_neg

    def get_smoothing_estimate(self, number_of_values):
        """
        Returns a Laplace smoothing estimate if m_estimate is negative
        :param number_of_values:
        :return:
        """
        if self.m_estimate < 0:
            return number_of_values
        else:
            return self.m_estimate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A Naive-Bayes Classifier Implementation.")
    parser.add_argument('data_file_name')
    parser.add_argument('m_estimate', type=float)
    args = parser.parse_args()

    example_set = parse_c45(args.data_file_name, DATA_DIRECTORY)
    data_set = np.array(example_set.to_float())
    for feature in example_set.schema[1:-1]:
        if feature.type == 'NOMINAL':
            feature.values = tuple([feature.to_float(value) for value in feature.values])
    normalize(data_set, example_set.schema)
    results = NaiveBayes.solve(data_set, example_set.schema[1:-1], args.m_estimate)
    print_performance(results)