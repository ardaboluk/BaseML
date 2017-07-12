
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import extractMnist
import extractCover

# we should verify our implementation by comparing the results with 
# scikit-learn's logistic regression

# read the data
print()
print("Data is being read...")
print()
[data, labels] = extractMnist.readMnist("../../Datasets/mnist/", True, digitZero = 0, digitOne = 9)
#[data, labels] = extractCover.readCover("../../Datasets/covertype/", True, classZero = 1, classOne = 2)

# convert data to float64
data = np.float64(data)

# roc array
roc_curves = []
pr_curves = []

# standardizer
standard_scaler = StandardScaler()
# logistic regression model
model = LogisticRegression()
# stratified cross validation
skf = StratifiedKFold(n_splits = 10, shuffle = True)
# fold number
foldNum = 1
for train_index, test_index in skf.split(np.zeros(data.shape[1]), labels):
	
	# print fold number
	print("Fold", foldNum)
	print()	
	
	# standardize data
	print("Standardizing data")
	print()
	trainData = standard_scaler.fit_transform(data[:, train_index].T)
	testData = standard_scaler.transform(data[:, test_index].T)
	trainLabels = labels[train_index]
	testLabels = labels[test_index]

	# fit the model to the training data
	print("Fitting model to the training data")
	print("")
	model.fit(trainData, trainLabels)
	
	# predict the class labels
	print("Predicting test data")
	print()
	pred = model.predict_proba(testData)[:,1]
	
	# calculate roc curve
	print("Calculating ROC and PR curves")
	print()
	cur_roc = metrics.roc_curve(testLabels, pred)
	cur_pr = metrics.precision_recall_curve(testLabels, pred)
	roc_curves.append(np.column_stack((cur_roc[0], cur_roc[1])))
	pr_curves.append(np.column_stack((cur_pr[0], cur_pr[1])))
	
	# increase fold number
	foldNum += 1
	
# plot the roc curves
colorlist = ["blue", "blueviolet", "brown", "coral", "cadetblue", "chartreuse", "darkgoldenrod", "firebrick", "fuchsia", "indigo"]
print("Plotting ROC curves")
print()
plt.figure()
plt.title("All ROC Curves for Scikit LogReg")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
foldNum = 1
for (currentRoc, currentColor) in zip(roc_curves, colorlist):
	plt.plot(currentRoc[:,0], currentRoc[:,1], color = currentColor, label = ("Fold" + str(foldNum)))
	foldNum += 1
plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), "k--", label = "Baseline")
plt.legend(loc = 4)

print("Plotting PR curves")
print()
plt.figure()
plt.title("All PR Curves for Scikit LogReg")
plt.xlabel("Recall")
plt.ylabel("Precision")
foldNum = 1
for (currentPr, currentColor) in zip(pr_curves, colorlist):
	plt.plot(currentPr[:,1], currentPr[:,0], color = currentColor, label = ("Fold" + str(foldNum)))
	foldNum += 1
pr_baseline = np.sum(labels == 1) / (np.sum(labels == 1) + np.sum(labels == 0))
plt.plot(np.linspace(1, 0, 5), np.ones(5) * pr_baseline, "r--", label = "Baseline")
plt.legend(loc = 3)

plt.show()
