import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = np.count_nonzero(prediction * ground_truth) / np.count_nonzero(prediction)
    recall = np.count_nonzero(prediction * ground_truth) / (np.count_nonzero(prediction * ground_truth) + (
                1 - np.count_nonzero(prediction) - 1 + np.count_nonzero(prediction + ground_truth)))
    accuracy = 1 - np.count_nonzero(prediction ^ ground_truth) / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 1 - np.count_nonzero(prediction - ground_truth) / len(ground_truth)
