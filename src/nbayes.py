import argparse
import copy
from mldata import parse_c45
import numpy as np
import training
from utils import timing, print_performance

DATA_DIRECTORY = 'data/'


class NaiveBayes(object):

    def __init__(self, m_estimate):
        self.m_estimate = m_estimate

    @classmethod
    @timing
    def solve(cls, data, m_estimate):
        return training.train_async(data, 5, cls, m_estimate)

    def train_nominal_feature(self, feature_set, labels, acceptable_values, m_estimate):
        positive_bin = {v: 0 for v in acceptable_values}
        negative_bin = copy.deepcopy(positive_bin)

        num_pos_class, num_neg_class = 0.0, 0.0

        for feature_example, label in zip(feature_set, labels):
            if label > 0:
                positive_bin[feature_example] += 1
                num_pos_class += 1.0
            else:
                negative_bin[feature_example] += 1
                num_neg_class += 1.0

        m = self.get_smoothing_operator(m_estimate, len(acceptable_values))
        prior = 1 / float(len(acceptable_values))

        for value in acceptable_values:
            positive_bin[value] = (positive_bin[value] + m * prior) / (num_pos_class + m)
            negative_bin[value] = (negative_bin[value] + m * prior) / (num_neg_class + m)

        return positive_bin, negative_bin

    def train_continuous_feature(self, feature_set, labels):
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

        pos_variance = 0.01 if pos_variance < .01 else pos_variance
        neg_variance = 0.01 if neg_variance < .01 else neg_variance

        return locals()

    @staticmethod
    def get_smoothing_operator(m_estimate, number_of_values):
        if m_estimate < 0:
            return number_of_values
        else:
            return m_estimate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A Naive-Bayes Classifier Implementation.")
    parser.add_argument('data_file_name')
    parser.add_argument('m_estimate', type=float)
    args = parser.parse_args()

    example_set = parse_c45(args.data_file_name, DATA_DIRECTORY)
    data_set = np.array(example_set.to_float())
    results = NaiveBayes.solve(data_set, args.m_estimate)
    print_performance(results)