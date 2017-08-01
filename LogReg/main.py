
import matplotlib.pyplot as plt
import numpy as np
import bgd
import folds
import evaluate

import sys
sys.path.insert(0, '../util')
import extractMnist
import extractCover
sys.path.pop(0)

# determine options for batch gradient descent
options = {"maxEpochs":200, "learningRate":0.15, "minCostDiff":0.001, "debug":False, "debugVerbose":False}

# read the data
print()
print("Data is being read...")
print()
[data, labels] = extractMnist.readMnist("../../../Datasets/mnist/", True, digitZero = 0, digitOne = 9)
#[data, labels] = extractCover.readCover("../../../Datasets/covertype/", True, classZero = 1, classOne = 2)

# show an image from the MNIST (if you desired)
#extractMnist.show(data[:,96], 28, 28)

# create folds for k-fold cross validation
print("Creating folds for cross validation...")
print()
numFolds = 10
folded_data, folded_labels = folds.generateFolds(data, labels, numFolds, verbose = True)

# cross validation
print("Evaluating the model using cross-validation")
print()
gradientDescent = bgd.BGD(options)
evaluator = evaluate.Evaluator()
roc_points = []
pr_points = []
for i in range(len(folded_labels)):
	
	print("Fold", i , "as test")
	
	# split train and test sets	
	if i == 0:
		train_data = np.hstack(folded_data[1:numFolds+1])
		train_labels = np.hstack(folded_labels[1:numFolds+1])
	elif i == numFolds - 1:
		train_data = np.hstack(folded_data[0:numFolds])
		train_labels = np.hstack(folded_labels[0:numFolds])
	else:
		train_data = np.concatenate((np.hstack(folded_data[0:i]), np.hstack(folded_data[i+1:numFolds+1])), axis = 1)
		train_labels = np.concatenate((np.hstack(folded_labels[0:i]), np.hstack(folded_labels[i+1:numFolds+1])))
		
	test_data = folded_data[i]
	test_labels = folded_labels[i]
	
	# standardize data
	# We should be careful not to contaminate the training data with any information about the test data.
	# Also, in practice we wouldn't have mean and std of the test data. Thus, we use the mean and std
	# of the training data to standardize both the training data and test data
	print("Train and test data is being normalized...")
	print()
	train_mean = np.mean(train_data, axis = 1)
	train_std = np.std(train_data, axis = 1) + 0.1
	trainMeanTrainTiled = np.tile(np.array([train_mean]).T, [1, train_data.shape[1]])
	trainStdTrainTiled = np.tile(np.array([train_std]).T, [1, train_data.shape[1]])
	trainMeanTestTiled = np.tile(np.array([train_mean]).T, [1, test_data.shape[1]])
	trainStdTestTiled = np.tile(np.array([train_std]).T, [1, test_data.shape[1]])
	
	train_data = (train_data - trainMeanTrainTiled) / trainStdTrainTiled
	test_data = (test_data - trainMeanTestTiled) / trainStdTestTiled
	
	# add 1s to train and test data as bias terms
	print("Adding bias terms to the train and the test data")
	print()
	train_data = np.vstack((train_data, np.ones(train_data.shape[1])))
	test_data = np.vstack((test_data, np.ones(test_data.shape[1])))
	
	# train the model on training data
	print("Training the model on the training data")
	print()
	optimizedTheta = gradientDescent.optimize(train_data, train_labels)
	
	# calculate roc and pr points
	print("Evaluating the model on the test data")
	print()
	roc_points_cur, pr_points_cur = evaluator.calc_rocpr(optimizedTheta, test_data, test_labels)
	roc_points.append(roc_points_cur)
	pr_points.append(pr_points_cur)

# calculate threshold-averaged roc and corresponding auroc
print("Calculating the threshold-averaged ROC and the corresponding AUC")
print()
rocAvgNumSamples = 200
prAvgNumSamples = 200
roc_avg = evaluator.calc_thr_avg_roc(roc_points, numSamples = rocAvgNumSamples)
pr_avg = evaluator.calc_thr_avg_pr(pr_points, numSamples = prAvgNumSamples)
auroc_avg = evaluator.calc_auroc(np.column_stack((roc_avg[:,0], roc_avg[:,2])))

print("AUC for the average ROC curve for", rocAvgNumSamples, "points:", auroc_avg)
print()

# get optimal threshold from the roc
print("Obtaining the optimal threshold from the ROC")
print()
p_pos = np.sum(labels == 1) / labels.shape[0]
p_neg = np.sum(labels == 0) / labels.shape[0]
roc_points_opt = np.column_stack((roc_avg[:,0], roc_avg[:,2], roc_avg[:,4], roc_avg[:,1], roc_avg[:,3], roc_avg[:,5]))
opt_roc_point = evaluator.get_opt_threshold_roc(roc_points_opt, p_pos, p_neg, 1, 1)

# Determining the optimal threshold on the test set is a mistake
# we may (and probably will) overfit the test data.
print("Optimal threshold obtained from ROC curve is:", opt_roc_point[2])
print()

colorlist = ["blue", "blueviolet", "brown", "coral", "cadetblue", "chartreuse", "darkgoldenrod", "firebrick", "fuchsia", "indigo"]
print("Plotting ROC curves")
print()
plt.figure()
plt.title("All ROC Curves for Custom LogReg")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
foldNum = 1
for (currentRoc, currentColor) in zip(roc_points, colorlist):
	plt.plot(currentRoc[:,0], currentRoc[:,1], color = currentColor, label = ("Fold" + str(foldNum)))
	foldNum += 1
plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), "k--", label = "Baseline")
plt.legend(loc = 4)

print("Plotting PR curves")
print()
plt.figure()
plt.title("All PR Curves for Custom LogReg")
plt.xlabel("Recall")
plt.ylabel("Precision")
foldNum = 1
for (currentPr, currentColor) in zip(pr_points, colorlist):
	plt.plot(currentPr[:,0], currentPr[:,1], color = currentColor, label = ("Fold" + str(foldNum)))
	foldNum += 1
pr_baseline = np.sum(labels == 1) / (np.sum(labels == 1) + np.sum(labels == 0))
plt.plot(np.linspace(1, 0, 5), np.ones(5) * pr_baseline, "r--", label = "Baseline")
plt.legend(loc = 3)

print("Plotting threshold-averaged ROC curve")
print()
plt.figure()
plt.title("Custom LogReg ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(roc_avg[:,0], roc_avg[:,2], "b-", label = "Logit")
# how to calculate confidence bands assuming binomial distribution ?
# Macskassy, S., & Provost, F. (2004). Confidence bands for ROC curves: Methods and an empirical study. 
# Proceedings of the First Workshop on ROC Analysis in AI. August 2004.
# as expected, the confidence bands become too large when the number of folds is low (10 e.g.)
#fpr_err = 1.96 * np.sqrt((roc_avg[:,0] * (1 - roc_avg[:,0])) / numFolds)
#tpr_err = 1.96 * np.sqrt((roc_avg[:,2] * (1 - roc_avg[:,2])) / numFolds)
#plt.errorbar(roc_avg[:,0], roc_avg[:,2], xerr = fpr_err, yerr = tpr_err)
# both tpr and fpr of a point on the baseline curve of ROC is equal to the frequency 
# with which the random classifier guesses the positive class
plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), "r--", label = "Baseline")
plt.plot(opt_roc_point[0], opt_roc_point[1], "ro")
plt.legend(loc = 4)

print("Plotting threshold-averaged PR curve")
print()
plt.figure()
plt.title("Custom LogReg PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(pr_avg[:,0], pr_avg[:,2], "b-", label = "Logit")
#recall_err = 1.96 * np.sqrt((pr_avg[:,0] * (1 - pr_avg[:,0])) / numFolds)
#prec_err = 1.96 * np.sqrt((pr_avg[:,2] * (1 - pr_avg[:,2])) / numFolds)
#plt.errorbar(pr_avg[:,0], pr_avg[:,2], xerr = recall_err, yerr = prec_err)
# precision is constant no matter what the frequency with which the random classifier guesses 
# the positive class and is equal to P / (P + N), but recall is equal to the frequency itself
pr_baseline = np.sum(labels == 1) / (np.sum(labels == 1) + np.sum(labels == 0))
plt.plot(np.linspace(1, 0, 5), np.ones(5) * pr_baseline, "r--", label = "Baseline")
plt.legend(loc = 3)

plt.show()
