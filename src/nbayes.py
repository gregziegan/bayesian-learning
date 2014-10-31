import argparse
from mldata import parse_c45
import numpy as np

DATA_DIRECTORY = 'data/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A Naive-Bayes Classifier Implementation.")
    parser.add_argument('data_file_name')
    args = parser.parse_args()

    example_set = parse_c45(args.data_file_name, DATA_DIRECTORY)
    data_set = np.array(example_set.to_float())