
import numpy as np

def generateFolds(data, labels, numFolds, verbose):
	"Class for separating data into folds using k-fold stratified cross-validation"
	
	# ratio of positive examples
	ratioPositive = np.sum(labels == 1) / labels.shape[0]
	
	# number of positive and negative examples in each fold
	numPosExFold = np.int64(np.ceil(ratioPositive * (labels.shape[0] / numFolds)))
	numNegExFold = np.int64(np.ceil(labels.shape[0] / numFolds) - numPosExFold)
	
	# indices of positive and negative examples
	posInd = np.where(labels == 1)[0]
	negInd = np.where(labels == 0)[0]
	
	# shuffle the indices
	np.random.shuffle(posInd)
	np.random.shuffle(negInd)
	
	# folds are stored in two lists as data and labels
	folds_data = []
	folds_labels = []
	posIndStart = 0
	negIndStart = 0
	posIndEnd = numPosExFold
	negIndEnd = numNegExFold
	
	# create the folds
	for foldNum in range(numFolds):
		
		# collect data
		currentFoldData = np.concatenate((data[:, posInd[posIndStart : posIndEnd]], data[:, negInd[negIndStart : negIndEnd]]), axis = 1)
		currentFoldLabels = np.concatenate((labels[posInd[posIndStart : posIndEnd]], labels[negInd[negIndStart : negIndEnd]]))
		
		# shuffle the current fold (otherwise positive examples would come first)
		currentFoldInd = np.arange(currentFoldLabels.shape[0]) 
		np.random.shuffle(currentFoldInd)
		
		# append the current fold to the list
		folds_data.append(currentFoldData[:,currentFoldInd])
		folds_labels.append(currentFoldLabels[currentFoldInd])
		
		# update the start and end positions of the indices
		posIndStart = posIndEnd
		posIndEnd = posIndStart + numPosExFold
		negIndStart = negIndEnd
		negIndEnd = negIndStart + numNegExFold
		
		if posIndEnd > posInd.shape[0]:
			posIndEnd = posInd.shape[0]
		
		if negIndEnd > negInd.shape[0]:
			negIndEnd = negInd.shape[0]
	
	# print information about each fold if requested
	if verbose == True:
		
		print("Total number of examples:", labels.shape[0])
		print("Total number positive examples:", np.sum(labels == 1))
		print("Total number of negative examples:", np.sum(labels == 0))
		print("Positive - Negative ratio (P/N):", np.sum(labels == 1) / np.sum(labels == 0))
		print()
		print("Fold", "# Examples", "# Pos Examples", "# Neg Examples", sep = "\t")
		
		for foldNum in range(numFolds):
			
			currentFoldData = folds_data[foldNum]
			currentFoldLabels = folds_labels[foldNum]
			print(foldNum, currentFoldLabels.shape[0], np.sum(currentFoldLabels == 1), np.sum(currentFoldLabels == 0), sep = "\t\t")
		
		print()
		
	return folds_data, folds_labels
