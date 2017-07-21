
import numpy as np
from extractMnistMulticlass import readMnist
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
    for curEpoch in range(0, maxEpochs):
        
        z = np.exp(W.dot(trainInputs))
        zSum = np.tile(np.sum(z, axis = 0), (trainTargets.shape[0], 1))
        p = np.divide(z, zSum)
        cost = - np.sum(np.multiply(np.log(p), trainTargets))
        gradients = - (trainTargets - p).dot(trainInputs.T)

        W -= alpha * gradients

        costDiff = abs(cost - prevCost)

        print("Epoch {} Cost {} Cost Diff {}".format(curEpoch + 1, cost, costDiff))

        if costDiff < minCostDiff:
            print("Cost difference is below threshold... stopping.")
            break
        else:
            prevCost = cost

    return W
    
# determine the parameters
num_folds = 10    

# read the dataset and convert labels to one-hot
mnist_data = readMnist("../../../Datasets/mnist")

train_data = mnist_data[0]
train_labels = mnist_data[1].reshape(-1,1)
test_data = mnist_data[2]
test_labels = mnist_data[3].reshape(-1,1)

# normalize data
train_data = preprocessing.scale(train_data, axis = 1)
test_data = preprocessing.scale(test_data, axis = 1)

enc = preprocessing.OneHotEncoder()
enc.fit(train_labels)
train_labels = enc.transform(train_labels).toarray().T
enc.fit(test_labels)
test_labels = enc.transform(test_labels).toarray().T

s = np.arange(train_labels.shape[1])
np.random.shuffle(s)
train_data = train_data[:, s]
train_labels = train_labels[:, s]
s = np.arange(test_labels.shape[1])
np.random.shuffle(s)
test_data = test_data[:, s]
test_labels = test_labels[:, s]

# append 1s to features of train inputs and test inputs for the bias term
train_data = np.vstack((train_data, np.ones((train_data.shape[1]))))
test_data = np.vstack((test_data, np.ones((test_data.shape[1]))))

W = trainSoftmax(train_data, train_labels)

# evaluate and calculate accruracy
predictions = W.dot(test_data)
acc = np.sum(np.argmax(test_labels, axis = 0) == np.argmax(predictions, axis = 0)) / test_labels.shape[1]
print("Accuracy: {}".format(acc))
