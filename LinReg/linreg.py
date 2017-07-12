
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt

def readData(data_loc, delim=","):
    """"Reads the dataset from the given location and returns it as a numpy array.
    Data is stroed as column-major matrix, that is, features are in rows."""

    data = np.genfromtxt(data_loc, delimiter=delim)[1:,:]

    return data.T

def generateFoldsReg(data, numFolds, foldNum):
    """Separates data into folds for k-fold cross validation. 
    Not all folds are returned. foldNum states the fold for test, starting from 1.
    Folds are not generated in a stratified manner. Both trainData and testData are column-major matrices."""

    folds = np.array_split(data, numFolds, axis = 1)
    testData = folds[foldNum - 1]
    trainFoldInds = np.delete(np.arange(numFolds), foldNum - 1)
    trainData = np.concatenate(itemgetter(*trainFoldInds)(folds), axis = 1)

    return trainData, testData



def trainLinReg(trainInputs, trainTargets, alpha = 0.05, maxEpochs = 1000, minCostDiff = 0.001):
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

numFolds = 10
alpha = 12e-10 # for non-standardized
#alpha = 76e-5 # for standardized
maxEpochs = 10000
minCostDiff = 0.0001
data = readData("../../Datasets/concrete/concrete.csv", delim=",")

verbose = True
normalize = False

# shuffle the data
tData = data.T
np.random.shuffle(tData)
data = tData.T

avgMSE = 0

for foldNum in range(numFolds):

    print("Fold {}".format(foldNum))

    # separate data into train and test partitions for the current fold
    trainData, testData = generateFoldsReg(data, numFolds, foldNum + 1)

    if normalize == True:
        # normalize data using the statistics of the training data
        train_mean = np.mean(trainData, axis = 1)
        train_std = np.std(trainData, axis = 1) + 0.1
        trainMeanTrainTiled = np.tile(np.array([train_mean]).T, [1, trainData.shape[1]])
        trainStdTrainTiled = np.tile(np.array([train_std]).T, [1, trainData.shape[1]])
        trainMeanTestTiled = np.tile(np.array([train_mean]).T, [1, testData.shape[1]])
        trainStdTestTiled = np.tile(np.array([train_std]).T, [1, testData.shape[1]])
        
        trainData = (trainData - trainMeanTrainTiled) / trainStdTrainTiled
        testData = (testData - trainMeanTestTiled) / trainStdTestTiled

    trainInputs = trainData[0:trainData.shape[0] - 1, :]
    trainTargets = trainData[trainData.shape[0] - 1, :]
    testInputs = testData[0:testData.shape[0] - 1, :]
    testTargets = testData[testData.shape[0] - 1, :]
    
    # append 1s to features of train inputs and test inputs for the bias term
    trainInputs = np.vstack((trainInputs, np.ones((trainInputs.shape[1]))))
    testInputs = np.vstack((testInputs, np.ones((testInputs.shape[1]))))

    # train the linear regression model on the train data
    W = trainLinReg(trainInputs, trainTargets, alpha, maxEpochs, minCostDiff)

    # test the model on test data and calculate MSE
    zTest = W.T.dot(testInputs)
    
    if normalize == True:
        
        # un-standardize target values
        zTest = np.multiply(zTest, train_std[8]) + train_mean[8]
        testTargets = np.multiply(testTargets, train_std[8]) + train_mean[8]
        
    MSETest = np.sum(np.power((zTest - testTargets), 2)) / testInputs.shape[1]

    if verbose == True:
        
        sortedTargetIndices = np.argsort(testTargets)
        plt.scatter(np.arange(testTargets.shape[0]), testTargets[sortedTargetIndices], color="r", label="actual")
        plt.scatter(np.arange(zTest.shape[1]), zTest[0, sortedTargetIndices], color="b", label="predicted")
        plt.legend()
        plt.show()

    print("MSE for fold {} on test data is {}".format(foldNum, MSETest))
    print()

    avgMSE += MSETest

avgMSE /= numFolds

print("Average MSE is {}".format(avgMSE))
