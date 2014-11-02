import argparse
from math import exp, log, pi
from mldata import parse_c45
import numpy as np
import os
import training
from utils import timing, print_performance

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '../data/')


class NaiveBayes(object):

    def __init__(self, m_estimate):
        self.m_estimate = m_estimate

        self._prob_pos_y = 0.0
        self._prob_neg_y = 0.0
        self._tree_nodes = []

    @classmethod
    @timing
    def solve(cls, data, schema, m_estimate):
        return training.train_async(data, schema, 5, cls, m_estimate)

    def train(self, examples, class_labels, schema):
        num_pos_labels = 0
        for label in class_labels:
            num_pos_labels += 1 if label > 0 else 0

        total_labels = float(len(class_labels))

        self._prob_pos_y = num_pos_labels / total_labels
        self._prob_neg_y = (total_labels - num_pos_labels) / total_labels

        for feature_index, feature in enumerate(schema):
            feature_set = [example[feature_index] for example in examples]

            if feature.type == 'CONTINUOUS':
                node = self.train_continuous_feature(feature_set, class_labels)
            else:
                node = self.train_nominal_feature(feature_set, class_labels, feature.values, self.m_estimate)

            self._tree_nodes.append(node)

    def classify(self, validation_data):
        return map(self.predict_example, validation_data)

    def predict_example(self, example):
        pos_prob = log(self._prob_pos_y)
        neg_prob = log(self._prob_neg_y)

        for feature_index, feature_value in enumerate(example):
            get_conditional_prob_func = self._tree_nodes[feature_index]['get_conditional_prob']
            conditional_prob_pos, conditional_prob_neg = get_conditional_prob_func(feature_value)

            if conditional_prob_pos > 0:
                pos_prob += log(conditional_prob_pos)

            if neg_prob > 0:
                neg_prob += log(conditional_prob_neg)

        estimate = pos_prob > neg_prob
        certainty = pos_prob

        return estimate, certainty

    @staticmethod
    def train_nominal_feature(feature_set, labels, nominal_values, m_estimate):
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

        m = NaiveBayes.get_smoothing_estimate(m_estimate, len(nominal_values))
        prior = 1 / float(len(nominal_values))

        for nominal_value in nominal_values:
            pos_bin[nominal_value] = (pos_bin[nominal_value] + m * prior) / (num_pos_class + m)
            neg_bin[nominal_value] = (neg_bin[nominal_value] + m * prior) / (num_neg_class + m)

        summary = {
            'pos_bin': pos_bin,
            'neg_bin': neg_bin,
            'get_conditional_prob': lambda example: (pos_bin[example], neg_bin[example])
        }

        return summary

    @staticmethod
    def train_continuous_feature(feature_set, labels):
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

        summary = locals()
        summary['get_conditional_prob'] = lambda e: NaiveBayes.get_continuous_conditional_probability(e, summary)
        return summary

    @staticmethod
    def get_continuous_conditional_probability(feature_value, summary):
        pos_mu, neg_mu = summary['pos_mean'], summary['neg_mean']
        pos_sig2, neg_sig2 = summary['pos_variance'], summary['neg_variance']
        prob_pos = 1 / (2 * pi * pos_sig2)**0.5 * exp(-0.5 * (feature_value - pos_mu)**2 / pos_sig2)
        prob_neg = 1 / (2 * pi * neg_sig2)**0.5 * exp(-0.5 * (feature_value - neg_mu)**2 / neg_sig2)
        return prob_pos, prob_neg

    @staticmethod
    def get_smoothing_estimate(m_estimate, number_of_values):
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
    for feature in example_set.schema[1:-1]:
        feature.values = tuple([feature.to_float(value) for value in feature.values])
    results = NaiveBayes.solve(data_set, example_set.schema[1:-1], args.m_estimate)
    print_performance(results)