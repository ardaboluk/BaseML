
import numpy as np
from extractMnistMulticlass import readMnist
from model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt
    
def trainSoftmax(trainInputs, trainTargets, alpha = 12e-8, maxEpochs = 1000, minCostDiff = 0.001):
    """Trains the softmax regression model with the given training data and returns model parameters.
    Training is stopped if maxEpochs is reached or the difference between subsequent costs is under minCostDiff."""

    # weight vector including the bias term
    W = np.zeros((trainTargets.shape[0], trainInputs.shape[0]))

    # prevCost as a stopping criterion
    prevCost = 0

    # epochs
    cost = 0
    costDiff = 0
    for curEpoch in range(0, maxEpochs):
        
        z = np.exp(W.dot(trainInputs))
        zSum = np.tile(np.sum(z, axis = 0), (trainTargets.shape[0], 1))
        p = np.divide(z, zSum)
        cost = - np.sum(np.multiply(np.log(p), trainTargets))
        gradients = - (trainTargets - p).dot(trainInputs.T)

        W -= alpha * gradients

        costDiff = abs(cost - prevCost)

        #print("Epoch {} Cost {} Cost Diff {}".format(curEpoch + 1, cost, costDiff))

        if costDiff < minCostDiff:
            print("Cost difference is below threshold... stopping.")
            break
        else:
            prevCost = cost

    print("Softmax final cost {} final cost diff {}".format(cost, costDiff))
            
    return W
    
# determine the parameters
numFolds = 10    

# read the dataset and convert labels to one-hot
mnist_data = readMnist("../../../Datasets/mnist")
data = np.concatenate((mnist_data[0], mnist_data[2]), axis = 1)
labels = np.concatenate((mnist_data[1], mnist_data[3])).reshape(-1,1)

# one-hot encoded labels
enc = preprocessing.OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray().T

# shuffle data and labels
s = np.arange(data.shape[1])
np.random.shuffle(s)
data = data[:, s].astype(float)
labels = labels[:, s]

# train and test in a k-fold stratified cross-validation setting
avg_acc = 0
stf = StratifiedKFold(labels, num_folds = numFolds)
curFold = 1
for train_indices, test_indices in stf:

    train_data = data[:, train_indices]
    train_labels = labels[:, train_indices]
    test_data = data[:, test_indices]
    test_labels = labels[:, test_indices]

    # normalize data
    std_scale = preprocessing.StandardScaler().fit(train_data.T)
    train_data = std_scale.transform(train_data.T).T
    test_data = std_scale.transform(test_data.T).T

    # append 1s to features of train inputs and test inputs for the bias term
    train_data = np.vstack((train_data, np.ones((train_data.shape[1]))))
    test_data = np.vstack((test_data, np.ones((test_data.shape[1]))))

    # train the softmax algorithm
    W = trainSoftmax(train_data, train_labels, maxEpochs = 10)

    # evaluate and calculate accruracy
    predictions = W.dot(test_data)
    acc = np.sum(np.argmax(test_labels, axis = 0) == np.argmax(predictions, axis = 0)) / test_labels.shape[1]
    avg_acc += acc
    print("Fold {} test accuracy: {}".format(curFold, acc))
    print()
    curFold += 1

avg_acc /= numFolds
print("Average accuracy {}".format(avg_acc))
