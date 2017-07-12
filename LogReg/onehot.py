
import numpy as np


def encodeOneHot(labels):
    "This function encodes given labels as one-hot vectors"

    # one-hot labels
    one_hot_labels = np.zeros((labels.shape[0], np.unique(labels).shape[0]));

    labelCounter = 0
    for curLabel in labels:

        one_hot_labels[labelCounter, labels[labelCounter]] = 1
        labelCounter += 1

    return one_hot_labels
                       
