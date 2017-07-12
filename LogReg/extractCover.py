
import numpy as np
import pandas as pd

def readCover(data_loc, binary_classes, classZero = 1, classOne = 2):
	"Reads data from the Forest Cover Type dataset and returns data as arrays"
	
	# data_loc - the location of the folder containing the dataset files
	# binary_classes - should be True if data is used for binary classification
	# classZero - one of the classes in binary classification, default value is 1
	# classOne - one of the classes in binary classification, default value is 2
	
	data = pd.read_csv(data_loc + "covtype.data").as_matrix()
	
	if binary_classes == True:
		data = data[np.logical_or(data[:,54] == classZero, data[:,54] == classOne),:]
	
	labels = data[:, 54]
	data = data[:, 0:54]
	
	if binary_classes == True:
		labels[labels == classZero] = 0
		labels[labels == classOne] = 1
	
	return [data.T, labels]
