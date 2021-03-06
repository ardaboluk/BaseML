import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sn
import itertools

import sys

sys.path.insert(0, '../util')
from extractMnist import readMnist
from model_selection import StratifiedKFold
import extractCover

import mlp

sys.path.pop(0)

# determine the parameters
numFolds = 7

# read the dataset and convert labels to one-hot

# mnist dataset
mnist_data = readMnist("../../../Datasets/mnist", binary_digits = False)
data = np.concatenate((mnist_data[0], mnist_data[2]), axis = 1)
labels = np.concatenate((mnist_data[1], mnist_data[3])).reshape(-1,1)

# cover type dataset
#[data, labels] = extractCover.readCover('../../../Datasets/covertype/', binary_classes=False)
#labels = labels.reshape(-1, 1)

# one-hot encoded labels
enc = preprocessing.OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray().T

# shuffle data and labels
s = np.arange(data.shape[1])
np.random.shuffle(s)
data = data[:, s].astype(float)
labels = labels[:, s]

# average accuracy
avg_acc = 0
# all of the predictions and true labels for calculating the final metrics
# we shouldn'd collect the metrics themselves and average them, because in k-fold cross validation,
# each sample is tested once
whole_true_labels = None
whole_predictions = None

# train and test in a k-fold stratified cross-validation setting
stf = StratifiedKFold(labels, num_folds=numFolds)
curFold = 1
training_losses = []
# MLP parameters
# lambda is weight decay parameter
dropout_keep_probs = [1., 1.]
options = {"sizeLayers": [data.shape[0] + 1, 128, labels.shape[0]], "dropouts": dropout_keep_probs,
           "maxEpochs":30, "learningRate":1e-3, "lambda":0.04, "batchSize": 100}
# define the mlp and construct the graph
mlp_4_layer = mlp.MLP(options)
mlp_4_layer.constructTrainingGraph(num_classes = labels.shape[0], num_features = data.shape[0] + 1)
mlp_4_layer.constructPredictionsGraph()

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

    # train the mlp algorithm
    # mini-batch gradient descent made a HUGE difference
    # it doesn't trying to directly converge to a local minimum, instead jumps over multiple local minimums and, hopefully, moves towards the global minimum 
    # also, in batch gradient descent, gradients are accumulated over the whole dataset, thus very sensitive to the learning rate    
    training_losses.append(mlp_4_layer.trainMLP(train_data, train_labels))

    # print the training accuracy
    train_predictions = mlp_4_layer.predict(train_data)
    train_acc = np.sum(np.argmax(train_labels, axis=0) == np.argmax(train_predictions, axis=0)) / train_labels.shape[1]
    print("Fold {} train accuracy: {}".format(curFold, train_acc))

    # evaluate the model on the test data
    predictions = mlp_4_layer.predict(test_data)

    # collect the predictions and true labels for the final metrics
    if whole_predictions is None:
        whole_predictions = predictions
        whole_true_labels = test_labels
    else:
        whole_predictions = np.concatenate((whole_predictions, predictions), axis=1)
        whole_true_labels = np.concatenate((whole_true_labels, test_labels), axis=1)

    # calculate accuracy
    acc = np.sum(np.argmax(test_labels, axis=0) == np.argmax(predictions, axis=0)) / test_labels.shape[1]
    avg_acc += acc
    print("Fold {} test accuracy: {}".format(curFold, acc))
    print()
    curFold += 1

# calculate and print average accuracy
avg_acc /= numFolds
print("Average test accuracy {}".format(avg_acc))

# plot training losses
plt.figure()
for i in range(len(training_losses)):
    plt.plot(np.arange(start = 1, stop = len(training_losses[i]) + 1), training_losses[i], label = "Fold {}".format(i))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend(loc = 1)
plt.show()


# calculate and plot confusion matrix
conf_mat = metrics.confusion_matrix(np.argmax(whole_true_labels, axis=0), np.argmax(whole_predictions, axis=0))
df_cm = pd.DataFrame(conf_mat, index=[i for i in range(0, test_labels.shape[0])],
                     columns=[i for i in range(0, test_labels.shape[0])])
plt.figure()
sn.heatmap(df_cm, annot=True)
plt.show()

# calculate and plot precision-recall roc curves for each class'
plt.figure(1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curve for all classes")
plt.figure(2)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve for all classes")
for curClass, (preds, trues, plot_marker) in enumerate(
        zip(whole_predictions, whole_true_labels, itertools.cycle(['-', '--', '-.', ':']))):
    cur_precision, cur_recall, _ = metrics.precision_recall_curve(trues, preds)
    cur_fpr, cur_tpr, _ = metrics.roc_curve(trues, preds)

    plt.figure(1)
    plt.plot(cur_recall, cur_precision, plot_marker, label="Class {}".format(curClass))
    plt.figure(2)
    plt.plot(cur_fpr, cur_tpr, plot_marker, label="Class {}".format(curClass))

plt.figure(1)
plt.legend(loc=3)
plt.figure(2)
plt.legend(loc=4)
plt.show()
