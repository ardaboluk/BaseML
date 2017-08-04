
from PIL import Image
import struct
import numpy as np

def readMnist(data_loc, binary_digits, digitZero = 0, digitOne = 1):
	"Reads images from the MNIST dataset and returns data as arrays"
	
	# data_loc - the location of the folder containing the dataset files
	# binary_digits - should be True if data is used for binary classification
	# digitZero - one of the digits in binary classification, default value is 0
	# digitOne - one of the digits in binary classification, default value is 1
	
	# reference: github/akesling/mnist.py
	
	# MNIST stores pixels in row-major order
	
	if data_loc[-1] != "/":
		data_loc += "/"
	
	# addresses of the data files
	train_data_loc = data_loc + "train-images.idx3-ubyte"
	train_labels_loc = data_loc + "train-labels.idx1-ubyte"
	test_data_loc = data_loc + "t10k-images.idx3-ubyte"
	test_labels_loc = data_loc + "t10k-labels.idx1-ubyte"

	# read files into arrays
	with open(train_data_loc, "rb") as train_data_file:
		magic, num, rows, cols = struct.unpack(">IIII", train_data_file.read(16))
		train_data = np.fromfile(train_data_file, dtype = np.uint8).reshape(rows * cols, num, order = "F")
	
	with open(test_data_loc, "rb") as test_data_file:
		magic, num, rows, cols = struct.unpack(">IIII", test_data_file.read(16))
		test_data = np.fromfile(test_data_file, dtype = np.uint8).reshape(rows * cols, num, order = "F")
		
	with open(train_labels_loc, "rb") as train_labels_file:
		magic,num = struct.unpack(">II", train_labels_file.read(8))
		train_labels = np.fromfile(train_labels_file, dtype = np.int8)
		
	with open(test_labels_loc, "rb") as test_labels_file:
		magic,num = struct.unpack(">II", test_labels_file.read(8))
		test_labels = np.fromfile(test_labels_file, dtype = np.int8)	
		
	# data for digits 1 and 0
	if binary_digits == True:
		train_data = train_data[:, np.logical_or(train_labels == digitZero, train_labels == digitOne)]
		test_data = test_data[:, np.logical_or(test_labels == digitZero, test_labels == digitOne)]
		train_labels = train_labels[np.logical_or(train_labels == digitZero, train_labels == digitOne)]
		test_labels = test_labels[np.logical_or(test_labels == digitZero, test_labels == digitOne)]
		
		# make labels 0 and 1 (logistic function gives values between 0 and 1)
		train_labels[train_labels == digitZero] = 0
		train_labels[train_labels == digitOne] = 1
		test_labels[test_labels == digitZero] = 0
		test_labels[test_labels == digitOne] = 1
		
	return [train_data, train_labels, test_data, test_labels]
	
	
def show(pixels, rows, cols):
	"Shows image whose data is given as vector"
	
	imageData = pixels.reshape(rows, cols)
	imageShown = Image.fromarray(imageData, "L")
	imageShown.show()
