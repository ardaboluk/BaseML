
import numpy as np

class StratifiedKFold:
    """Iterator class for k-fold stratified cross validation.
    Can be used for stratification in a multi-class setting on one-hot encoded labels.
    In labels matrix, rows are assumed to represent class labels and columns to represent samples."""

    def __init__(self, labels, num_folds = 10):

        self.__num_folds = num_folds

        self.__folds_indices = self.__split(labels)

        # determines which fold to be the test set when the caller requests the training and test indices
        self.__nextFoldInd = 0

    def __split(self, labels):
        """Divides the labels into folds and returns their indices. 
        This method is private as the user will call nextFoldIndices to get training and test indices for the next fold."""

        # indices of samples in each fold
        folds_indices = []

        num_classes = labels.shape[0]
        num_samples = labels.shape[1]

        # determine indices of samples in each fold
        for cur_class in range(0, num_classes):
            cur_class_folds = np.array_split(np.where(np.argmax(labels, axis = 0) == cur_class)[0], self.__num_folds)            
            for cur_class_cur_fold in range(0, self.__num_folds):
                if len(folds_indices) < cur_class_cur_fold + 1:
                    folds_indices.append(cur_class_folds[cur_class_cur_fold])
                else:
                    folds_indices[cur_class_cur_fold] = np.concatenate((folds_indices[cur_class_cur_fold], cur_class_folds[cur_class_cur_fold]))

        
        # shuffle indices in each fold, otherwise samples would be fed to the learner ordered by class
        for cur_fold in range(0, self.__num_folds):
            np.random.shuffle(folds_indices[cur_fold])

        return folds_indices

    def __iter__(self):
        return self

    def __next__(self):
        """This method returns the training and test indices for the next fold."""

        if self.__nextFoldInd == self.__num_folds:
            raise StopIteration

        test_indices = self.__folds_indices[self.__nextFoldInd]

        if self.__nextFoldInd == 0:
            train_indices = np.concatenate(self.__folds_indices[1:self.__num_folds])
        elif self.__nextFoldInd == self.__num_folds - 1:
            train_indices = np.concatenate(self.__folds_indices[0:self.__num_folds-1])
        else:        
            train_indices = np.concatenate((np.concatenate(self.__folds_indices[0:self.__nextFoldInd]), np.concatenate(self.__folds_indices[self.__nextFoldInd+1:self.__num_folds])))

        self.__nextFoldInd += 1

        return train_indices, test_indices
        
