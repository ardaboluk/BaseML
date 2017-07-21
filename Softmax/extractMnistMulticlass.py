
import struct
import numpy as np

def readMnist(data_loc):
        "Reads images from the MNIST dataset and returns data as arrays"
	
        # data_loc - the location of the folder containing the dataset files
	
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
                        
        return [train_data, train_labels, test_data, test_labels]
