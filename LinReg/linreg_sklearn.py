
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from linreg import readData

def main():

    numFolds = 10
    standardize = True
    verbose = True

    data_loc = "../../Datasets/concrete/concrete.csv"
    data = np.genfromtxt(data_loc, delimiter = ',')

    np.random.shuffle(data)

    kf = KFold(n_splits = numFolds)

    avgMSE = 0

    for train_index, test_index in kf.split(data):

        train_data, test_data = data[train_index], data[test_index]

        if standardize == True:
            scaler = StandardScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

        train_inputs, train_targets = train_data[:, :-1], train_data[:, -1]
        test_inputs, test_targets = test_data[:, :-1], test_data[:, -1]

        reg = LinearRegression().fit(train_inputs, train_targets)
        pred = reg.predict(test_inputs)

        # de-standardize predicted targets and test targets if standardization is performed on data before
        if standardize == True:
            std = np.power(scaler.var_, 0.5)
            pred = np.multiply(pred, std[8]) + scaler.mean_[8]
            test_targets = np.multiply(test_targets, std[8]) + scaler.mean_[8]

        MSETest = mean_squared_error(pred, test_targets)

        if verbose == True:
            
            sortedTargetIndices = np.argsort(test_targets)
            plt.scatter(np.arange(test_targets.shape[0]), test_targets[sortedTargetIndices], color="r", label="actual")
            plt.scatter(np.arange(pred.shape[0]), pred[sortedTargetIndices], color="b", label="predicted")
            plt.legend()
            plt.show()

        print("MSE for the current test data is {}".format(MSETest))
        print()

        avgMSE += MSETest
    
    avgMSE /= numFolds

    print("Average MSE is {}".format(avgMSE))

if __name__ == "__main__":

    main()
