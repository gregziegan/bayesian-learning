import time
import numpy as np


def get_random_initial_weights(examples):
    return np.random.rand(examples.shape[1]) * .01


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f seconds' % (f.func_name, (time2-time1))
        return ret
    return wrap


def get_accuracy(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):
    return (num_true_positives + num_true_negatives) / (num_true_positives + num_true_negatives +
                                                        num_false_positives + num_false_negatives)


def get_precision(num_true_positives, num_false_positives):
    if num_false_positives == 0:
        return 1.0
    return num_true_positives / (num_true_positives + num_false_positives)


def get_recall(num_true_positives, num_false_negatives):
    if num_false_negatives == 0:
        return 1.0
    return num_true_positives / (num_true_positives + num_false_negatives)


def normalize(data, schema):
    stds = data.std(axis=0)
    means = data.mean(axis=0)
    for example_index in range(0, len(data)):
        for i in range(1, data[example_index].size - 1):
            if schema.features[i].type == 'CONTINUOUS':
                data[example_index][i] = (data[example_index][i] - means[i]) / stds[i]
    return data


def print_performance(results):
    num_true_positives, num_false_positives, num_true_negatives, num_false_negatives = 0.0, 0.0, 0.0, 0.0

    accuracies, precisions, recalls = [], [], []
    for result in results:
        predictions, class_labels = result['predictions'], result['class_labels']
        for prediction_tuple, class_label in zip(predictions, class_labels):
            prediction, certainty = prediction_tuple
            if prediction > 0:
                if class_label > 0:
                    num_true_positives += 1
                else:
                    num_false_positives += 1
            else:
                if class_label <= 0:
                    num_true_negatives += 1
                else:
                    num_false_negatives += 1

            accuracies.append(get_accuracy(num_true_positives, num_false_positives,
                                           num_true_negatives, num_false_negatives))
            precisions.append(get_precision(num_true_positives, num_false_positives))
            recalls.append(get_recall(num_true_positives, num_false_negatives))

    print "Accuracy: {:0.3f} {:0.3f}".format(np.mean(accuracies), np.std(accuracies))
    print "Precision: {:0.3f} {:0.3f}".format(np.mean(precisions), np.std(precisions))
    print "Recall: {:0.3f} {:0.3f}".format(np.mean(recalls), np.std(recalls))