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


def contingency_table(result, threshold):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    predictions, class_labels = result['predictions'], result['class_labels']
    for prediction_tuple, class_label in zip(predictions, class_labels):
        confidence = prediction_tuple > threshold
        if class_label and confidence:
            tp += 1
        elif class_label:
            fn += 1
        elif confidence:
            fp += 1
        else:
            tn +=1
    return tp, fn, fp, tn

def print_performance(results):
    num_true_positives, num_false_positives, num_true_negatives, num_false_negatives = 0.0, 0.0, 0.0, 0.0
    roc = []
    rs = []
    tp = []
    tn = []
    fp = []
    fn = []
    fp_rate = [0.0]
    tp_rate = [0.0]
    accuracies, precisions, recalls = [], [], []

    for result in results:
        r = []
        predictions, class_labels = result['predictions'], result['class_labels']
        for prediction_tuple, class_label in zip(predictions, class_labels):
            prediction, certainty = prediction_tuple
            r.append(certainty)
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

            roc.extend(result)
            rs.extend(r)

            accuracies.append(get_accuracy(num_true_positives, num_false_positives,
                                           num_true_negatives, num_false_negatives))
            precisions.append(get_precision(num_true_positives, num_false_positives))
            recalls.append(get_recall(num_true_positives, num_false_negatives))


            rs, roc = zip(*sorted(zip(rs, roc), reverse=True))

            for r in rs:
                contingency_table(roc, r)
                if num_false_positives:
                    fp_rate.append(num_false_positives / float(num_false_positives + num_true_negatives))
                else:
                    fp_rate.append(0.0)
                if num_true_positives:
                    tp_rate.append(num_true_positives / float(num_true_positives + num_false_negatives))
                else:
                    tp_rate.append(0.0)

            aroc = 0
            for point_one, point_two in zip(zip(fp_rate[0:-1], tp_rate[0:-1]), zip(fp_rate[1:], tp_rate[1:])):
                aroc += ((point_two[0] - point_one[0]) * (point_two[1] + point_one[1])) / 2.0

    print "Accuracy: {:0.3f} {:0.3f}".format(np.mean(accuracies), np.std(accuracies))
    print "Precision: {:0.3f} {:0.3f}".format(np.mean(precisions), np.std(precisions))
    print "Recall: {:0.3f} {:0.3f}".format(np.mean(recalls), np.std(recalls))
    print "AROC: {:0.3f}".format(aroc)
