
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
	"Class for evaulating the performance of a given model"		

	def calc_rocpr(self, theta, X, y):
		"Classify the data for different cut-off values and return the results and the cutoff values"
		
		# number of features and instances
		numExamples = y.shape[0]
		numFeatures = X.shape[0]
		
		# calculate excitation
		z = theta.T.dot(X)
		sigma = 1 / (1 + np.exp(-z))
		
		# examples of label 0 assumed to be false
		numCondNegative = np.sum(y == 0)
		numCondPositive = np.sum(y.shape[0] - numCondNegative)
		
		# efficient calculation of ROC curves
		# Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition letters, 27(8), 861-874.
		sortedInd = np.argsort(-sigma, kind = "heapsort")
		fp = 0
		tp = 0
		cur_score = 1
		pr_points = []
		roc_points = []
		fprev = -2
		
		for currentInd in sortedInd:
			if sigma[currentInd] != fprev:
				
				roc_points.append([fp / numCondNegative, tp / numCondPositive, cur_score])
								
				if tp + fp != 0:
					pr_points.append([tp / numCondPositive, tp / (tp + fp), cur_score])
			
			if y[currentInd] == 1:
				tp += 1
			else:
				fp += 1
			
			cur_score = sigma[currentInd]
		
		roc_points.append([fp / numCondNegative, tp / numCondPositive, 0])
		pr_points.append([tp / numCondPositive, tp / (tp + fp), 0])
		
		return np.asarray(roc_points), np.asarray(pr_points)
		
	def calc_auroc(self, roc_points):
		"Calculate area under curve from given roc points"
		
		prev_fpr = roc_points[0][0]
		prev_tpr = roc_points[0][1]
		
		auroc = 0
		
		for i in range(len(roc_points) - 1):
			
			cur_fpr = roc_points[i + 1][0]
			cur_tpr = roc_points[i + 1][1]
			
			auroc += np.abs(cur_fpr - prev_fpr) * ((cur_tpr + prev_tpr) / 2)
			
			prev_fpr = cur_fpr
			prev_tpr = cur_tpr
			
		return auroc
		
	def calc_thr_avg_roc(self, roc_points_arr, numSamples):
		"Threshold-averages the given ROC curves"
		
		# numSamples is the number of threshold samples on which the curves will be averaged
		
		# array that contains averages of the roc points
		avg_array = []
		
		# all scores of all roc points
		allScores = []
		for curRoc in roc_points_arr:
			for curRow in curRoc:
				allScores.append(curRow[2])
				
		sortedScores = -np.sort(-np.asarray(allScores), kind = "heapsort")
		
		prev_roc_point_inds = np.int64(np.ones(len(roc_points_arr)))
		
		for scoreInd in np.int64(np.linspace(0, sortedScores.shape[0], num = numSamples, endpoint = False)):
			
			fprs = []
			tprs = []
			thrs = []
			curThreshold = sortedScores[scoreInd]
			
			for rocInd in range(len(roc_points_arr)):
				
				roc_point_ind = prev_roc_point_inds[rocInd]				
				while roc_point_ind < len(roc_points_arr[rocInd]) and roc_points_arr[rocInd][roc_point_ind][2] > curThreshold:					
					roc_point_ind += 1
				prev_roc_point_inds[rocInd] = roc_point_ind
				
				fprs.append(roc_points_arr[rocInd][roc_point_ind][0])
				tprs.append(roc_points_arr[rocInd][roc_point_ind][1])
				thrs.append(roc_points_arr[rocInd][roc_point_ind][2])
				
			avg_array.append([np.mean(fprs), np.std(fprs), np.mean(tprs), np.std(tprs), np.mean(thrs), np.std(thrs)])
			
		return np.asarray(avg_array)
		
	def calc_thr_avg_pr(self, pr_points_arr, numSamples):
		"Threshold-averages the given PR curves"
		
		# numPos and numNeg contains number of positive and negative examples in each fold
		
		# numSamples is the number of threshold samples on which the curves will be averaged
		
		# array that contains averages of the roc points
		avg_array = []
		
		# all scores of all roc points
		allScores = []
		for curRoc in pr_points_arr:
			for curRow in curRoc:
				allScores.append(curRow[2])
				
		sortedScores = -np.sort(-np.asarray(allScores), kind = "heapsort")
		
		prev_pr_point_inds = np.int64(np.ones(len(pr_points_arr)))
		
		for scoreInd in np.int64(np.linspace(0, sortedScores.shape[0], num = numSamples, endpoint = False)):
			
			tps = []
			fps = []
			tprs = []
			precs = []			
			curThreshold = sortedScores[scoreInd]
			
			for prInd in range(len(pr_points_arr)):
				
				pr_point_ind = prev_pr_point_inds[prInd]
				while pr_point_ind < len(pr_points_arr[prInd]) and pr_points_arr[prInd][pr_point_ind][2] > curThreshold:					
					pr_point_ind += 1
				prev_pr_point_inds[prInd] = pr_point_ind
				
				tprs.append(pr_points_arr[prInd][pr_point_ind][0])
				precs.append(pr_points_arr[prInd][pr_point_ind][1])
			
			avg_array.append([np.mean(tprs), np.std(tprs), np.mean(precs), np.std(precs)])
			
		return np.asarray(avg_array)
		
		
	def get_opt_threshold_roc(self, roc_points, p_pos, p_neg, c_fp, c_fn):
		"Returns the point on ROC curve giving the optimal threshold regarding the classification cost and the threshold itself"
		
		# Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. 
		# Machine learning, 42(3), 203-231.
		
		# roc_points is a numpy array of the form either [fp_rate, tp_rate, threshold] or [fp_rate, tp_rate, threshold, std_fpr, std_tpr, std_thr]
		# p_pos and p_neg represents ratio of positive and negative examples, respectively
		# c_fp and c_fn represent the costs of false positive and false negative classifications, respectively 
		# isavg should be true if threshold-averaged roc is used and roc_points should be of the second form
		
		minCost = 10e5
		minCostFpr = 0
		minCostTpr = 0
		minCostThr = 0
		
		for curRocPoint in roc_points:
			
			curFpr = curRocPoint[0]
			curTpr = curRocPoint[1]
			curCost = p_pos * (1 - curTpr) * c_fn + p_neg * curFpr * c_fp
			
			if curCost < minCost:
				
				minCost = curCost
				minCostFpr = curFpr
				minCostTpr = curTpr
				minCostThr = curRocPoint[2]
		
		return [minCostFpr, minCostTpr, minCostThr]
