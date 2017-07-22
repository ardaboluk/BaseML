
import numpy as np

class StratifiedKFold:
    """Can be used for stratification in a multi-class setting on one-hot encoded labels.
    In labels matrix, rows are assumed to represent class labels and columns to represent samples."""

    def __init__(self, labels, num_folds = 10):

        self.__num_folds = num_folds

    def __split(labels):
        """Determines train and test indices for each fold and returns them as two 2-d numpy arrays
        where each row of the first array represents indices for the training set and each row of the second
        array represent indices for the test set. This method is private as the user will call nextFoldIndices to get
        training and test indices for the next fold."""

        num_classes = labels.shape[0]
        num_samples = labels.shape[1]

        # indices of samples belonging to each class
        class_indices = []
        for cur_class in range(0, num_classes):
            class_indices.append(np.where(np.argmax(labels, axis = 0) == cur_class)[0])

        
        
        
        
