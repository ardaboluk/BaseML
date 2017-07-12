
import numpy as np
import unit
import evaluate as ev

class BGD:
	"Class for batch gradient descent"
	
	# it's assumed that rows represent features and columns represent examples
	# thus trainData[j,i] is the j-th feature of the i-th training example
	
	def __init__(self, options):
		"Constructor initializes parameters for bgd"
		
		# options is a dictionary datatype
		self.__maxEpochs = options["maxEpochs"]
		self.__learningRate = options["learningRate"]
		self.__minCostDiff = options["minCostDiff"]
		self.__debug = options["debug"]
		self.__debugVerbose = options["debugVerbose"]
		
	
	def optimize(self, X, y):
		"optimizes weights to learn the data"
		
		# number of examples and features
		numExamples = X.shape[1]
		numFeatures = X.shape[0]
		
		# initialize a neuron object
		neuron = unit.Neuron()
		
		# initialize all weights to 0 (include bias)
		theta = np.zeros(numFeatures)
		# initialize weights to a small number
		#theta = np.random.uniform(0.1, 0.9, numFeatures)
		
		# use previous cost as a stopping criterion
		prevCost = 0
		
		# begin epochs
		for currentEpoch in range(self.__maxEpochs):
			
			# calculate cost and gradients
			[currentCost, currentGradients] = neuron.calculate(theta, X, y)
			
			# check gradients if debug mode is on
			if self.__debug == True:
				print("Checking gradients...")
				print()
				neuron.gradientCheck(theta, X, y, currentGradients, self.__debugVerbose)
			
			# update weights	
			theta -= self.__learningRate * (1 / numExamples) * currentGradients
			
			# print epoch, prev cost and current cost
			print("Epoch:", currentEpoch + 1)
			print("Prev cost:", prevCost, "Current cost:", currentCost, "Difference:", abs(currentCost - prevCost))
			print()
			
			# stop if the difference between prevCost and currentCost is small enough
			if abs(currentCost - prevCost) < self.__minCostDiff:
				print("Change in cost smaller than threshold... stopping")
				print()
				break
							
			if currentEpoch + 1 == self.__maxEpochs:
				print("Reached the maximum number of epochs")
				print()
			
			# update prevCost
			prevCost = currentCost
		
		return theta
		
