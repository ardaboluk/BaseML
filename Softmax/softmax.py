
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from matplotlib.pyplot as plt

def readData(data_loc, delim=","):
    """"Reads the dataset from the given location and returns it as a numpy array.
    Data is stroed as column-major matrix, that is, features are in rows."""

    data = np.genfromtxt(data_loc, delimiter=delim)[1:,:]

    return data.T
    
def trainSoftmax(trainInputs, trainTargets, alpha = 0.05, maxEpochs = 1000, minCostDiff = 0.001):
    """Trains the linear regression model with the given training data and returns model parameters.
    trainData contains target values in the last row.
    Training is stopped if maxEpochs is reached or the difference between subsequent costs is under minCostDiff."""

    # weight vector including the bias term
    W = np.zeros((trainInputs.shape[0], 1)) + 0.01

    # prevCost as a stopping criterion
    prevCost = 0

    # epochs
    for curEpoch in range(0, maxEpochs):
        
        z = W.T.dot(trainInputs)
        cost = 0.5 * np.sum(np.power(z - trainTargets, 2))
        gradients = trainInputs.dot((z - trainTargets).T)

        W -= alpha * gradients

        costDiff = abs(cost - prevCost)

        if costDiff < minCostDiff:
            print("Cost difference is below threshold... stopping.")
            break
        else:
            prevCost = cost

    return W
    
# determine the parameters
num_folds = 10    

data = readData("../../Datasets/concrete/concrete.csv", delim = ",")

skf = StratifiedKFold(n_splits = num_folds, shuffle = True)

for train_index, test_index in skf.split(