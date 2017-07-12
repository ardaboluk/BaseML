
import numpy as np

class Neuron:
	"Class for one neuron unit"
		
	def calculate(self, theta, X, y):
		"Calculates the cost and the gradients"
		
		# theta - column vector storing the weights
		# X - the examples stored in a matrix
		#	- X[i,j] is the i-th feature of the j-th example
		# y - vector containing the labels
		
		numExamples = y.shape[0]
		numFeatures = X.shape[0]
		
		# z = total input to the neuron for each example
		# 1 x numExamples
		z = theta.T.dot(X)
		
		# sigma = 1 / (1 + exp(-z))
		# 1 x numExamples
		sigma = 1 / (1 + np.exp(-z))
		
		# total cost
		cost = -np.sum((y * np.log(sigma)) + ((1-y)*np.log(1-sigma)))
		
		# weight gradients
		gradients = X.dot((sigma - y).T)
		
		# return the cost and the gradients
		return [cost, gradients]
		
	def gradientCheck(self, theta, X, y, calc_grad, verbose):
		"Checks the gradients numerically"
		
		# calc_grad is calculated gradients
		
		# epsilon
		ep = 1e-4
		
		# sum of differences between calculated and approximated gradients
		sumDiff = 0
		
		# if verbose is true, difference between calculated and 
		# predicted values for each gradient is printed
		if verbose == True:
			print("Differences between the calculated and approximated gradients:")
			print()
		
		for index in range(theta.shape[0]):
			
			# epsilon vector
			currentEpVector = np.zeros(theta.shape[0])
			currentEpVector[index] = ep
			
			# calculate cost for theta - epsilon
			zMinus = (theta - currentEpVector).T.dot(X)
			sigmaMinus = 1 / (1 + np.exp(-zMinus))
			costMinus = -np.sum((y * np.log(sigmaMinus)) + ((1-y)*np.log(1-sigmaMinus)))
			
			# calculate cost for theta + epsilon
			zPlus = (theta + currentEpVector).T.dot(X)
			sigmaPlus = 1 / (1 + np.exp(-zPlus))
			costPlus = -np.sum((y * np.log(sigmaPlus)) + ((1-y)*np.log(1-sigmaPlus)))
			
			# approximated gradient
			apprxGrad = (costPlus - costMinus) / (2 * ep)
			
			# difference between the current calculated and the approximated gradient
			gradDiff = abs(apprxGrad - calc_grad[index])
			
			# sum the differences for average
			sumDiff += gradDiff
			
			# print the current difference
			if verbose == True:
				print("Gradient", index, ":", apprxGrad, calc_grad[index], gradDiff)
				print()
		
		print("Avg grad difference:", sumDiff / theta.shape[0])
		print()
		
		
