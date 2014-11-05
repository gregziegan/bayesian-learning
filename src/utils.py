import time
import numpy as np


def get_random_initial_weights(examples):
    random = np.random.RandomState(seed=12345)
    return random.rand(examples.shape[1]) * 0.1


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


def contingency_table(predictions, class_labels, threshold):
    tp, fn, fp, tn = 0, 0, 0, 0
    for prediction, class_label in zip(predictions, class_labels):
        confidence = prediction >= threshold
        if class_label == 1.0 and confidence:
            tp += 1
        elif class_label == 1.0:
            fn += 1
        elif confidence:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


@timing
def get_area_under_roc(predictions, class_labels):
    predictions, class_labels = zip(*sorted(zip(predictions, class_labels), reverse=True))
    fp_rate, tp_rate = [0.0], [0.0]

    for confidence in predictions:
        true_p, false_n, false_p, true_n = contingency_table(predictions, class_labels, confidence)
        if false_p:
            fp_rate.append(false_p / float(false_p + true_n))
        else:
            fp_rate.append(0.0)
        if true_p:
            tp_rate.append(true_p / float(true_p + false_n))
        else:
            tp_rate.append(0.0)

    aroc = 0.0
    for point_one, point_two in zip(zip(fp_rate[0:-1], tp_rate[0:-1]), zip(fp_rate[1:], tp_rate[1:])):
        aroc += ((point_two[0] - point_one[0]) * (point_two[1] + point_one[1])) / 2.0

    return aroc


def print_performance(results):
    num_true_positives, num_false_positives, num_true_negatives, num_false_negatives = 0.0, 0.0, 0.0, 0.0
    tp, tn, fp, fn = [], [], [], []
    accuracies, precisions, recalls = [], [], []

    ##  ROC calculation arguments ##
    all_predictions = []
    labels = []

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
            tp.append(num_true_positives)
            tn.append(num_true_negatives)
            fp.append(num_false_positives)
            fn.append(num_false_negatives)

            accuracies.append(get_accuracy(num_true_positives, num_false_positives,
                                           num_true_negatives, num_false_negatives))
            precisions.append(get_precision(num_true_positives, num_false_positives))
            recalls.append(get_recall(num_true_positives, num_false_negatives))

            # Below are lists used for ROC calculation
            all_predictions.append(certainty)
            labels.append(class_label)

    print "Accuracy: {:0.3f} {:0.3f}".format(np.mean(accuracies), np.std(accuracies))
    print "Precision: {:0.3f} {:0.3f}".format(np.mean(precisions), np.std(precisions))
    print "Recall: {:0.3f} {:0.3f}".format(np.mean(recalls), np.std(recalls))

    if len(all_predictions) > 10000:
        print 'Warning: this algorithm was run on a large data set.'
        print 'An AROC calculation will take n^2 time.'

        response = raw_input('Would you still like to calculate the area under the ROC curve? (Y/n): ')
        if response in ['Y', 'y']:
            print "AROC: {:0.3f}".format(get_area_under_roc(all_predictions, labels))